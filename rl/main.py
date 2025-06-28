# Standard library
import argparse

# New: import the updated helpers

from rl.model_loader import load_llama2code
from rl.data_loader import (
    get_apps_dataloader,
)
from rl.evaluate import evaluate

# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------

def _verify(device: str = "cuda", skip_download: bool = False) -> None:
    """Download (if necessary) and load the model to ensure the checkpoint is
    valid and all tensors can be materialised on *device*.
    """

    _ = load_llama2code(device=device, skip_download=skip_download)


def _evaluate(device: str = "cuda", skip_download: bool = False, batch_size: int = 1, split: str = "test") -> None:
    """Run evaluation over the APPS *test* split using the provided model.

    The evaluation logic is delegated to :pyfunc:`rl.evaluate.evaluate` – once
    that function is implemented it will be executed here.
    """

    # Build the model – this will also perform verification implicitly.
    model = load_llama2code(device=device, skip_download=skip_download)

    # Fetch ready-made DataLoader and tokenizer.
    dataloader, tokenizer = get_apps_dataloader(split=split, batch_size=batch_size)

    # Delegate to the (to-be-implemented) evaluation routine.
    evaluate(model, dataloader, tokenizer, device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main entry point for the Adversarial-RL project."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify -----------------------------------------------------------------
    verify_parser = subparsers.add_parser(
        "verify", help="Download + load the model to verify weights."
    )
    verify_parser.add_argument(
        "--device", default="cuda", help="Target device (e.g. 'cuda', 'cpu')."
    )
    verify_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume weights are already present locally.",
    )

    # evaluate ---------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run evaluation on the APPS test split."
    )
    eval_parser.add_argument(
        "--device", default="cuda", help="Target device (e.g. 'cuda', 'cpu')."
    )
    eval_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume weights are already present locally.",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the evaluation DataLoader.",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "verify":
        _verify(device=args.device, skip_download=args.skip_download)
    elif args.command == "evaluate":
        _evaluate(
            device=args.device,
            skip_download=args.skip_download,
            batch_size=args.batch_size,
            split=args.split,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
