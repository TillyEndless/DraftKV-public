# SPDX-License-Identifier: Apache-2.0
"""Slots-only unit-batched DMA; immutable Dense executor is untouched.

The ABI comes from CUDA's public driver header. Older driver/toolkit installs
retain the original producer. PyTorch's old CUPTI may omit batch transfers;
use a modern Nsight trace with copyCount for activity and bandwidth reporting.
"""
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
from functools import lru_cache
from pathlib import Path
import ctypes, hashlib, os, re

@lru_cache(None)
def batch_include():
    from torch.utils.cpp_extension import CUDA_HOME
    roots = [Path(CUDA_HOME)] if CUDA_HOME else []
    roots.extend(Path('/usr/local').glob('cuda-*'))
    candidates = []
    for root in roots:
        for include in (root / 'include', root / 'targets/x86_64-linux/include'):
            header = include / 'cuda.h'
            if not header.is_file():
                continue
            match = re.search('^#define CUDA_VERSION\\s+(\\d+)', header.read_text(), re.M)
            if match and int(match[1]) >= 12080:
                candidates.append((int(match[1]), str(include)))
    return min(candidates)[1] if candidates else None

def available(device=None):
    import torch
    if torch.cuda.get_device_capability(device) != (9, 0):
        return False
    version = ctypes.c_int()
    rc = ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(version))
    return rc == 0 and version.value >= 12080 and (batch_include() is not None)

def make_producer(device):
    if not available(device):
        return None
    return native().AsyncProducer(True)

@lru_cache(None)
def native():
    raise RuntimeError('EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED: CUDA 12.8+ Hopper slot_batch_cuda/binding.cpp has no recovered source fingerprint; this conditional producer is not released or runtime verified')
