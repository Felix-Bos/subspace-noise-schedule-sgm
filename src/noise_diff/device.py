import os
import random

import numpy as np
import torch

######################
### Device selection
######################
"""
Picks the best available accelerator: CUDA GPU, then Apple Silicon GPU (MPS), then CPU.
The NOISE_DIFF_DEVICE environment variable (e.g. "cpu") overrides the automatic choice.
"""


def get_device(preferred=None):
    preferred = preferred or os.environ.get("NOISE_DIFF_DEVICE")
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    if device.type == "cuda":
        return f"CUDA GPU ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return f"CPU ({torch.get_num_threads()} threads)"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
