"""Helpers shared by the stages that need PyTorch (training and export)."""

from __future__ import annotations

import sys


def require_training_libraries() -> None:
    """Exit with install instructions if torch or transformers is missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"This step needs the packages in requirements-train.txt ({exc}).\n"
            f"Install them with: {sys.executable} -m pip install --only-binary=:all: "
            "-r requirements-train.txt"
        ) from None


def pick_device() -> str:
    """Return the accelerator PyTorch will use: cuda, mps or cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
