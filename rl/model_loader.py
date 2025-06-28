"""
Download (if needed) and load the un-quantised Code Llama-7B model.

Requirements:
    pip install torch safetensors huggingface_hub tqdm
"""
from pathlib import Path
from typing import Dict
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download           # HF ≥ 0.18.0
from rl.model import LlamaCode2

H = LlamaCode2.H
NUM_LAYERS = LlamaCode2.LAYERS
ID = LlamaCode2.ID


LOCAL_DIR: Path = Path("CodeLlama-7b")         
# only pull the weight shards + their index and the config
ALLOW_PATTERNS = [
    "pytorch_model*bin", "pytorch_model*.safetensors*",
    "*model*.index.json", "config.json"
]
# ---------------------------------------------------------------------


def download_if_missing() -> Path:
    """Download all weight shards the first time we run."""
    if any(LOCAL_DIR.glob("pytorch_model*")):      # already there
        print(f"✓ weights found in {LOCAL_DIR}")
        return LOCAL_DIR

    print(f"Downloading {ID} into {LOCAL_DIR} …")
    snapshot_download(
        repo_id=ID,
        local_dir=str(LOCAL_DIR),
        allow_patterns=ALLOW_PATTERNS,
        resume_download=True,                      # idempotent
    )
    print("✓ download complete")
    return LOCAL_DIR


def load_state_dict_to_device(model_dir: Path, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load shards directly to specified device to avoid CPU memory bottleneck."""
    def _load_to_device(shard: Path):
        if shard.suffix == ".safetensors":
            # SafeTensors supports direct device loading
            return load_file(str(shard), device=device)
        else:
            return torch.load(shard, map_location=device)

    state: Dict[str, torch.Tensor] = {}
    #  The glob also returns the index JSON file (e.g. pytorch_model.bin.index.json) which is not a tensor
    #  and will raise an error when passed to torch.load. Filter to keep only valid weight files.
    shards = [p for p in sorted(model_dir.glob("pytorch_model*")) if p.suffix in (".bin", ".safetensors")]
    if not shards:
        raise FileNotFoundError("No shards found – did the download fail?")
    
    for shard in shards:
        print(f"→ reading {shard.name} to {device}")
        state.update(_load_to_device(shard))
    return state


def sanity_check(state: Dict[str, torch.Tensor]) -> None:
    """Quickly confirm a few key tensor shapes."""
    for layer in range(NUM_LAYERS):
        key = f"model.layers.{layer}.self_attn.q_proj.weight"
        assert state[key].shape == (H, H), (
            f"{key} has shape {state[key].shape}, expected {(H, H)}"
        )
    print("✓ all checked shapes match spec")


# ------------------------------------------------------------------
#                      Mapping to our implementation
# ------------------------------------------------------------------


def _remap_key(hf_key: str) -> str | None:
    """Translate a single HF checkpoint key to the equivalent key in
    our `LlamaCode2` implementation. Keys that do not have a counterpart
    (e.g. rotary embeddings, multiple adapter heads, etc.) return None
    and will be skipped.

    Example mappings:
        model.embed_tokens.weight -> embeddings.weight
        model.layers.0.self_attn.q_proj.weight -> blocks.0.attention.q_proj.weight
        model.layers.0.mlp.gate_proj.weight    -> blocks.0.mlp.gate_proj.weight
    """

    if hf_key == "model.embed_tokens.weight":
        return "embeddings.weight"
    if hf_key == "model.norm.weight":
        return "norm.weight"
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    # Layer-wise keys
    prefix = "model.layers."
    if hf_key.startswith(prefix):
        remainder = hf_key[len(prefix):]  # e.g. "0.self_attn.q_proj.weight"
        layer_idx, sub_key = remainder.split(".", 1)

        # Attention projections ------------------------------------------------
        if sub_key.startswith("self_attn."):
            att_sub = sub_key[len("self_attn."):]  # q_proj.weight, etc.
            att_map = {
                "q_proj.weight": "attention.q_proj.weight",
                "k_proj.weight": "attention.k_proj.weight",
                "v_proj.weight": "attention.v_proj.weight",
                "o_proj.weight": "attention.out_proj.weight",
            }
            if att_sub in att_map:
                return f"blocks.{layer_idx}.{att_map[att_sub]}"

        # MLP projections ------------------------------------------------------
        if sub_key.startswith("mlp."):
            # gate_proj.weight, ...
            mlp_sub = sub_key[len("mlp."):]  # e.g. gate_proj.weight
            return f"blocks.{layer_idx}.mlp.{mlp_sub}"

        # LayerNorm weights ----------------------------------------------------
        if sub_key == "input_layernorm.weight":
            return f"blocks.{layer_idx}.input_layernorm.weight"
        if sub_key == "post_attention_layernorm.weight":
            return f"blocks.{layer_idx}.post_attention_layernorm.weight"

    # Unhandled key
    return None


def _convert_and_remap_state_dict(hf_state: Dict[str, torch.Tensor], *, dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
    """Create a new state-dict whose keys match our implementation and whose
    tensors are converted to ``dtype`` (default: bfloat16).
    Any weights that do not have a mapping will be silently dropped.
    """

    new_state: Dict[str, torch.Tensor] = {}
    skipped = 0
    for k, v in hf_state.items():
        new_k = _remap_key(k)
        if new_k is None:
            skipped += 1
            continue
        new_state[new_k] = v.to(dtype)

    if skipped:
        print(f"⚠️  skipped {skipped} HF weights with no mapping to local model")
    return new_state


def load_llamacode2(device: str = "cuda", skip_download: bool = False) -> LlamaCode2:
    """High-level convenience: download shards (if necessary), load them to
    ``device``, remap & cast to bf16 and finally return a fully initialised
    ``LlamaCode2`` instance ready for inference.
    """

    model_dir = LOCAL_DIR if skip_download else download_if_missing()

    # Step 1: load HF shards on *CPU* irrespective of the target device to avoid
    # holding two full copies of the weights on the GPU. This requires enough RAM
    # but prevents CUDA OOM when the model parameters are later materialised.
    hf_state = load_state_dict_to_device(model_dir, device="cpu")

    # Decide on the dtype for the final model – stay in bf16 on CPU, else use fp16
    target_dtype = torch.float16 if device.startswith("cuda") else torch.bfloat16

    print("→ remapping + casting weights …")
    state_dict = _convert_and_remap_state_dict(hf_state, dtype=target_dtype)

    # We no longer need the original HF state on CPU; clear it to free memory
    del hf_state

    # Step 2: instantiate the model on CPU, load weights, then move to the target device.
    model = LlamaCode2().to(dtype=target_dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Free the remapped state dict ASAP to release RAM
    del state_dict

    # Finally move the fully initialised model to the requested device.
    model = model.to(device)

    if missing:
        print(f"⚠️  {len(missing)} local parameters were *not* initialised from checkpoint → e.g. {missing[:3]}")
    if unexpected:
        print(f"⚠️  {len(unexpected)} checkpoint weights had no destination in the model → e.g. {unexpected[:3]}")

    print("✓ model loaded successfully")
    return model