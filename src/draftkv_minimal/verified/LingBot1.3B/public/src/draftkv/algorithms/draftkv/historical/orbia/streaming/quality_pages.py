# SPDX-License-Identifier: Apache-2.0
"""Quality-only pinned storage with bounded observational overhead.

The tensor operations and lifetime match original HostPages('pytorch'). Page
size/NUMA diagnostics are deliberately unmeasured, never inferred or cached.
Systems/THP runs continue to use the original full mapping evidence.
"""
from __future__ import annotations
from draftkv.runtime import profiling as pm
from draftkv.streaming.external import external_symbol, external_path
import time

class QualityPages:

    def __init__(self):
        self.allocations = []
        self.evidence = []

    def copy(self, source):
        if source.device.type != 'cpu' or source.numel() == 0:
            raise ValueError('HostPages.copy requires a nonempty CPU tensor')
        source = source.contiguous()
        started = pm.clock_ns()
        tensor = source.pin_memory()
        if not tensor.is_pinned():
            raise RuntimeError('quality host storage is not CUDA-pinned')
        self.allocations.append(tensor)
        if pm.enabled(): self.evidence.append({'payload_bytes': source.numel() * source.element_size(), 'post_register': None, 'pinned_verified': True, 'mapping_evidence': 'NOT_COLLECTED_QUALITY_ONLY', 'setup_ms': (pm.clock_ns() - started) / 1000000.0})
        return tensor

    def close(self):
        import torch
        torch.cuda.synchronize()
