"""Quality-only pinned storage with bounded observational overhead.

The tensor operations and lifetime match original HostPages('pytorch'). Page
size/NUMA diagnostics are deliberately unmeasured, never inferred or cached.
Systems/THP runs continue to use the original full mapping evidence.
"""
import time


class QualityPages:
    def __init__(self):
        self.allocations = []
        self.evidence = []

    def copy(self, source):
        if source.device.type != 'cpu' or source.numel() == 0:
            raise ValueError('HostPages.copy requires a nonempty CPU tensor')
        source = source.contiguous()
        started = time.perf_counter_ns()
        tensor = source.pin_memory()
        if not tensor.is_pinned():
            raise RuntimeError('quality host storage is not CUDA-pinned')
        self.allocations.append(tensor)
        self.evidence.append({
            'payload_bytes': source.numel() * source.element_size(),
            'post_register': None,
            'pinned_verified': True,
            'mapping_evidence': 'NOT_COLLECTED_QUALITY_ONLY',
            'setup_ms': (time.perf_counter_ns() - started) / 1e6,
        })
        return tensor

    def close(self):
        import torch
        # Same original synchronization/lifetime boundary: no reuse before DMA
        # and consumer completion. PyTorch owns unregistration and its allocator.
        torch.cuda.synchronize()
