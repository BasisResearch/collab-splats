"""General-purpose PyTorch and model-loading utilities."""

import gc
from typing import Any, Generator, List

import torch
from huggingface_hub import hf_hub_download


########################################################
########## Device helpers ##############################
########################################################


def get_device() -> str:
    """Return 'cuda' if a CUDA device is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def pytorch_gc():
    """Clear CUDA cache and run Python garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


########################################################
########## Batching ####################################
########################################################


def infer_batch_size(mem_per_image_gb: float, headroom: float = 0.3) -> int:
    """Compute safe batch size from available VRAM.

    Args:
        mem_per_image_gb: Estimated GPU memory per image in GB.
        headroom: Fraction of VRAM reserved for batch data (rest = model weights).

    Returns:
        Batch size >= 1. Returns 1 if CUDA is unavailable.
    """
    if mem_per_image_gb <= 0:
        raise ValueError(f"mem_per_image_gb must be positive, got {mem_per_image_gb}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return max(1, int(vram_gb * headroom / mem_per_image_gb))
    return 1


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """Yield aligned batches of size *batch_size* from one or more parallel sequences.

    Args:
        batch_size: Number of items per batch.
        *args: One or more sequences of equal length.

    Yields:
        A list of sliced sequences, one per input arg.
    """
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args), (
        "Batched iteration must have inputs of all the same size."
    )
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


########################################################
########## Model loading ###############################
########################################################


def load_hf_weights(repo_id: str, filename: str):
    """Download a single file from a Hugging Face Hub repository.

    Args:
        repo_id: HuggingFace repo (e.g. ``"facebook/dinov2-small"``).
        filename: File path within the repo.

    Returns:
        Local path to the downloaded file.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_torchhub_model(repo_id: str, model_name: str):
    """Load a pre-trained model from torch.hub.

    Args:
        repo_id: GitHub repo (e.g. ``"facebookresearch/dinov2"``).
        model_name: Entry-point name registered in the repo's ``hubconf.py``.

    Returns:
        The loaded model (nn.Module).
    """
    return torch.hub.load(repo_id, model_name)


########################################################
########## Registry mixin ##############################
########################################################


class RegistryMixin:
    """Name-based class registry. Declare ``_registry: Dict[str, type] = {}`` in subclass."""

    _registry: dict

    @classmethod
    def register(cls, name: str):
        """Class decorator: register a subclass under *name*."""
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str):
        """Return registered class for *name*, or raise ValueError."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
