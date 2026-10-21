"""
PyTorch device, precision, batching and model-loading helpers.

- RegistryMixin: name-to-class registry shared by the backend and model zoos
"""

import gc
import logging
import sys
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

########################################################
########## Device helpers ##############################
########################################################


def get_device() -> str:
    """
    Torch device name for this process.

    Returns:
        "cuda" when a CUDA device is available, else "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def pytorch_gc() -> None:
    """
    Run Python garbage collection, then release freed CUDA blocks to the driver.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Matmul precision is process-wide; one lock orders every change against a running forward
_matmul_precision_lock = threading.RLock()


@contextmanager
def full_fp32_matmul() -> Iterator[None]:
    """
    Run the block at full fp32 matmul precision, then restore the caller's setting.

    - TF32, which a mapanything import enables, would round the matmul
    - the setting is process-wide: waits while another thread holds hold_matmul_precision

    Yields:
        None; the block runs at "highest" precision.
    """
    with _matmul_precision_lock:
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

        try:
            yield
        finally:
            torch.set_float32_matmul_precision(precision)


@contextmanager
def hold_matmul_precision() -> Iterator[None]:
    """
    Keep the matmul precision unchanged for the block, across threads.

    - full_fp32_matmul on another thread waits until the block exits
    - wrap a forward that runs while other threads do guarded matmuls

    Yields:
        None.
    """
    with _matmul_precision_lock:
        yield


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Host numpy view of a tensor or array.

    - bf16 has no numpy dtype: cast to float32 first
    - ndarrays pass through uncopied

    Args:
        x: tensor on any device, or an ndarray.

    Returns:
        The data as an ndarray.
    """
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


@contextmanager
def vendored_path(root: Path, hint: str) -> Iterator[None]:
    """
    Vendored checkout on sys.path for the block, first unless already present.

    - missing checkout: ImportError naming the fix, before any import runs
    - exit removes the entry only if this call inserted it

    Args:
        root: checkout directory to import from.
        hint: how to obtain the checkout, appended to the error.

    Yields:
        Nothing; imports inside the block resolve against root.
    """
    if not root.is_dir():
        raise ImportError(f"{root} not found — {hint}")

    # Insert once; leave a caller's pre-existing entry alone
    entry = str(root)
    inserted = entry not in sys.path
    if inserted:
        sys.path.insert(0, entry)

    try:
        yield
    finally:
        if inserted and entry in sys.path:
            sys.path.remove(entry)


########################################################
########## Batching ####################################
########################################################


def infer_batch_size(mem_per_image_gb: float, headroom: float = 0.3) -> int:
    """
    Safe batch size from the device's total VRAM.

    Args:
        mem_per_image_gb: estimated GPU memory per image, in GB.
        headroom: fraction of VRAM reserved for batch data (the rest for model weights).

    Returns:
        Batch size >= 1; 1 when CUDA is unavailable.

    Raises:
        ValueError: if mem_per_image_gb is not positive.
    """
    if mem_per_image_gb <= 0:
        raise ValueError(f"mem_per_image_gb must be positive, got {mem_per_image_gb}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return max(1, int(vram_gb * headroom / mem_per_image_gb))
    return 1


def batch_iterator(batch_size: int, *args: Any) -> Generator[list[Any], None, None]:
    """
    Aligned batches of batch_size items from one or more parallel sequences.

    Args:
        batch_size: items per batch; the last batch may be shorter.
        *args: one or more sequences of equal length.

    Yields:
        One slice per input sequence, in argument order.

    Raises:
        ValueError: no sequences, or sequences of different lengths.
    """
    if not args or any(len(a) != len(args[0]) for a in args):
        raise ValueError(
            "batch_iterator needs one or more sequences of the same length"
        )
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


########################################################
########## Model loading ###############################
########################################################


def load_hf_weights(repo_id: str, filename: str) -> str:
    """
    Download one file from a Hugging Face Hub repository.

    Args:
        repo_id: Hub repo, e.g. "facebook/dinov2-small".
        filename: file path within the repo.

    Returns:
        Local path to the downloaded (or cached) file.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_torchhub_model(repo_id: str, model_name: str) -> torch.nn.Module:
    """
    Pre-trained model from torch.hub.

    Args:
        repo_id: GitHub repo, e.g. "facebookresearch/dinov2".
        model_name: entry point registered in the repo's hubconf.py.

    Returns:
        The loaded model.
    """
    return torch.hub.load(repo_id, model_name)


########################################################
########## Registry mixin ##############################
########################################################


class RegistryMixin:
    """
    Name-based class registry.

    - each registry base declares its own `_registry: ClassVar[dict[str, type]] = {}`
    """

    _registry: ClassVar[dict]

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        """
        Class decorator registering a subclass under a name.

        Args:
            name: registry key, e.g. the config backend name.

        Returns:
            A decorator that registers the class and returns it unchanged.
        """

        def decorator(subclass: type) -> type:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """
        Registered class for a name.

        Args:
            name: registry key.

        Returns:
            The class registered under name.

        Raises:
            ValueError: if name is not registered; the message lists the known names.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
