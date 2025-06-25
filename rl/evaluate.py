import torch
from torch.utils.data import DataLoader
import subprocess
import tempfile
import os
from contextlib import suppress
from typing import Union



def evaluate(model: torch.nn.Module, dataloader: DataLoader, tokenizer, device: Union[str, torch.device]):
    model.eval()

    # ------------------------------------------------------------------
    # Generation hyper-parameters
    # ------------------------------------------------------------------
    # Meta recommends using a temperature of 0.2 for Code-Llama-7B-Instruct
    TEMPERATURE = 0.2
    MAX_NEW_TOKENS = 512  # more than enough for most APPS solutions

    # Selecting a batch-size
    # ---------------------
    # A g6.xlarge instance has an NVIDIA L4 GPU with 24 GB of VRAM.
    # In BF16, the 7-B parameter model occupies ≈13 GB leaving ample room
    # for the KV-cache, activations and a few prompt / generated tokens.
    # Empirically a batch-size of *two* comfortably fits in memory while
    # still providing a throughput benefit over purely sequential eval.
    # The DataLoader constructed in ``rl.main`` already respects a user
    # supplied ``batch_size`` – we merely sanity-check that it does not
    # exceed *2* and warn otherwise.

    BATCH_LIMIT = 2

    dataloader_batch_size = dataloader.batch_size
    if dataloader_batch_size and dataloader_batch_size > BATCH_LIMIT:
        print(
            f"⚠️  Requested batch-size {dataloader_batch_size} exceeds the safe limit of {BATCH_LIMIT} on a g6.xlarge; results may OOM."
        )

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer is missing an eos_token_id – cannot perform autoregressive generation.")

    def _sample_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Take unnormalised logits and return sampled token-IDs (one per row)."""

        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate(
        input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """Very small, self-contained autoregressive generator that works with
        *any* causal LM that returns full-sequence logits. It is **not** optimised
        for speed – sufficient for one-pass evaluation.
        """

        model_input_ids = input_ids.clone()
        model_attention = attention_mask.clone().bool()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(model_input_ids, attention_mask=model_attention)
                next_token_logits = logits[:, -1, :]
                next_token = _sample_next_token(next_token_logits, temperature=TEMPERATURE)

                # Break if every sample hit EOS → all finished
                if (next_token == eos_id).all():
                    model_input_ids = torch.cat([model_input_ids, next_token.unsqueeze(1)], dim=1)
                    break

                # Append new token and extend mask
                model_input_ids = torch.cat([model_input_ids, next_token.unsqueeze(1)], dim=1)
                model_attention = torch.cat(
                    [model_attention, torch.ones_like(next_token, dtype=torch.bool).unsqueeze(1)], dim=1
                )

        return model_input_ids

    def _run_code(source: str, _input: str, timeout: int = 5) -> tuple[int, str, str]:
        """Run *source* code in a fresh Python subprocess feeding *_input* to
        STDIN. Returns ``(return_code, stdout, stderr)``.
        """

        # Use a temporary file so the full solution can be executed even if
        # it contains quotes / newlines that would break a `python -c` call.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(source)
            tmp_path = tmp.name

        try:
            completed = subprocess.run(
                ["python", tmp_path],
                input=_input.encode(),
                capture_output=True,
                timeout=timeout,
            )
            return completed.returncode, completed.stdout.decode(), completed.stderr.decode()
        finally:
            with suppress(FileNotFoundError):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    total, correct = 0, 0

    for batch in dataloader:
        # Prepare KV-cache for this batch and move tensors to target device
        model.set_kv_cache(batch["input_ids"].size(0))

        # Move tensors to target device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Autoregressive generation
        full_sequences = generate(input_ids, attention_mask, max_new_tokens=MAX_NEW_TOKENS)

        # Detach & move back to CPU for decoding
        full_sequences = full_sequences.cpu()

        # Decode *only* the newly generated portion per sample
        generated_codes = []
        for i in range(full_sequences.size(0)):
            prompt_len = attention_mask[i].sum().item()
            gen_ids = full_sequences[i, prompt_len:]
            # Stop decoding at EOS if present
            if (gen_ids == eos_id).any():
                eos_index = (gen_ids == eos_id).nonzero(as_tuple=False)[0].item()
                gen_ids = gen_ids[:eos_index]
            generated_codes.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

        # Evaluate each generated programme against its test-cases
        for code, tc_list in zip(generated_codes, batch["test_cases"]):
            total += 1
            success = True
            for _input, expected_out in tc_list:
                rc, stdout, _ = _run_code(code, _input)
                if rc != 0 or stdout.strip() != expected_out.strip():
                    success = False
                    break
            if success:
                correct += 1

        print(f"Processed {total} / ? samples — running accuracy: {correct}/{total} ({correct/total:.2%})", end="\r")

    # Final report
    print("\n✓ Evaluation complete")
    print(f"Final accuracy: {correct} / {total} = {correct/total:.2%}")