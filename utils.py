import random
import numpy as np
import torch

from config import RANDOM_SEED


def seed_everything(seed: int = RANDOM_SEED):
    """
    Set all random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Returns the best available device.

    Priority:
        CUDA -> MPS -> CPU
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"\nUsing device: {device}")

    return device


def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )