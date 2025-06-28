from typing import Optional, Tuple

from datasets import load_dataset, DatasetDict, Dataset

import json
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import DataLoader

# Model hyper-parameters
from rl.model import LlamaCode2

ID = LlamaCode2.ID
MAX_LENGTH = LlamaCode2.MAX_LENGTH

# ----------------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------------

def load_tokenizer(model_id: str = ID) -> PreTrainedTokenizerBase:
    """Load the Hugging-Face tokenizer used for Code Llama.«"""

    return AutoTokenizer.from_pretrained(model_id, use_fast=False)


# no longer returns DatasetDict directly – instead returns the *processed* split
# together with the tokenizer so that the caller can build a DataLoader.

def get_apps_dataset(
    split: str = "train", *, cache_dir: Optional[str] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> Tuple[Dataset, PreTrainedTokenizerBase]:
    """Return the APPS *split* with prompts tokenised and test-cases parsed.

    Each element is a ``dict`` with the following keys::

        {
            "input_ids": LongTensor         # (L,)
            "attention_mask": LongTensor    # (L,)
            "test_cases": List[Tuple[str, str]]
            "problem_id": int
            "difficulty": str
        }

    The dataset is set to *torch* format for the tensor columns so it can be
    consumed directly by ``torch.utils.data.DataLoader``.
    """

    if tokenizer is None:
        tokenizer = load_tokenizer()

    raw_ds = load_dataset("codeparrot/apps", split=split, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # STEP 1 – filter out prompts whose tokenised length exceeds context.
    # ------------------------------------------------------------------

    def _within_ctx_len(example):
        # tokenise *once* without truncation so we can measure the true length
        ids = tokenizer(example["question"], add_special_tokens=False)["input_ids"]
        return len(ids) <= MAX_LENGTH

    raw_ds = raw_ds.filter(_within_ctx_len)

    def _process(example):
        # ---------- prompt → tokens -----------------------------------------
        tokens = tokenizer(
            example["question"].strip(),
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # ---------- input / output pairs ------------------------------------
        io_pairs = json.loads(example["input_output"])
        test_cases = list(zip(io_pairs["inputs"], io_pairs["outputs"]))

        return {
            "input_ids": tokens["input_ids"][0],       # drop batch dim
            "attention_mask": tokens["attention_mask"][0],
            "test_cases": test_cases,
            "problem_id": example["problem_id"],
            "difficulty": example["difficulty"],
        }

    ds = raw_ds.map(_process, remove_columns=raw_ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds, tokenizer


# ----------------------------------------------------------------------------
# Collation – turns a *list* of processed examples into a single batch dict.
# ----------------------------------------------------------------------------

def _collate(batch, pad_id: int):
    """Collate function that pads variable-length sequences and leaves the
    Python objects (``test_cases``, ``problem_id``) untouched.
    """

    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "test_cases": [b["test_cases"] for b in batch],
        "problem_ids": [b["problem_id"] for b in batch],
        "difficulties": [b["difficulty"] for b in batch],
    }


def make_apps_collate_fn(tokenizer: PreTrainedTokenizerBase):
    """Return a ``collate_fn`` bound to a specific *tokenizer* (needed for the
    correct ``pad_token_id``).
    """

    return partial(_collate, pad_id=tokenizer.pad_token_id)


def get_apps_dataloader(
    *,
    split: str = "train",
    batch_size: int = 1,
    cache_dir: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, PreTrainedTokenizerBase]:
    """Return a ``torch.utils.data.DataLoader`` ready for iteration along with
    the ``tokenizer`` that was used to create it. This hides the collate
    mechanics from high-level code (e.g. ``rl.main``).
    """

    dataset, tokenizer = get_apps_dataset(split=split, cache_dir=cache_dir, tokenizer=tokenizer)
    collate_fn = make_apps_collate_fn(tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader, tokenizer
