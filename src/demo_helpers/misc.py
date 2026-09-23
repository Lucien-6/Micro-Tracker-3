#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from time import perf_counter
import torch


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class PeriodicVRAMReport:
    """Simpler helper object used to periodically check on VRAM usage (with cuda only)"""

    def __init__(self, update_period_ms=1000):

        self._has_cuda = torch.cuda.is_available()
        self._update_period_ms = round(update_period_ms)
        self._next_update_time_ms = 0
        self._vram_usage_mb = None

    def get_vram_usage(self):

        # Bail if cuda isn't being used
        if not self._has_cuda:
            return None

        # Update recorded VRAM usage periodically
        curr_time_ms = int(1000 * perf_counter())
        if curr_time_ms > self._next_update_time_ms:
            self._vram_usage_mb = get_total_cuda_vram_usage_mb()
            self._next_update_time_ms = curr_time_ms + self._update_period_ms

        return self._vram_usage_mb


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_default_device_string():
    """Helper used to select a default device depending on available hardware"""

    # Figure out which device we can use
    default_device = "cpu"
    if torch.backends.mps.is_available():
        default_device = "mps"
    if torch.cuda.is_available():
        default_device = "cuda"

    return default_device


SAM2_NATIVE_SIDE = 1024
SAM3_NATIVE_SIDE = 1008
SAM3_SIDE_MULTIPLE = 336
DEFAULT_ENCODE_SIDE = 1344


def native_encode_side(model_family: str) -> int:
    """Native longest side for an explicit -b. SAM 2 is 1024; SAM 3 and 3.1 are 1008."""

    if model_family == "v2":
        return SAM2_NATIVE_SIDE
    return SAM3_NATIVE_SIDE


def resolve_encode_side(explicit_side: int | None) -> int:
    """Side used when encoding. An omitted -b stays at 1344 until an A/B comparison decides otherwise."""

    if explicit_side is None:
        return DEFAULT_ENCODE_SIDE
    return int(explicit_side)


def encode_side_warning(model_family: str, side: int) -> str | None:
    """SAM 3 window alignment warning. None when the side needs no warning."""

    if model_family in ("v3", "v3p1") and int(side) % SAM3_SIDE_MULTIPLE != 0:
        return f"SAM 3 encode side {int(side)} is not a multiple of {SAM3_SIDE_MULTIPLE}."
    return None


def make_device_config(device_str, use_float32, use_channels_last=True, prefer_bfloat16=True):
    """Helper used to construct a dict for device usage. Meant to be used with 'model.to(**config)'"""

    f16_type = torch.bfloat16 if prefer_bfloat16 else torch.float16
    fallback_dtype = torch.float32 if (device_str == "cpu") else f16_type
    dtype = torch.float32 if use_float32 else fallback_dtype
    memory_format = torch.channels_last if use_channels_last else None

    return {"device": device_str, "dtype": dtype, "memory_format": memory_format}


def normalize_to_npuint8(tensor_data):
    """Function used to normalize tensor data to a 0-255 numpy uint8 format, meant for displaying as image data!"""
    min_val, max_val = tensor_data.min(), tensor_data.max()
    data_norm = (tensor_data - min_val) / (max_val - min_val)
    return (data_norm * 255).byte().cpu().numpy()


def get_total_cuda_vram_usage_mb():
    """Helper used to measure the total VRAM usage when using CUDA. Returns 0 if not using CUDA"""
    if torch.cuda.is_available():
        free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
        return (total_vram_bytes - free_vram_bytes) // 1_000_000
    return 0
