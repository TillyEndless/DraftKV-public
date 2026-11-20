"""Registered immutable pinned pages for Clean Commit; base4k by default.

Host memory is owned for the entire worker lifetime: CUDA registration must
outlive the views read by async B01 producer DMA. No pages are reused.
"""
from __future__ import annotations

import mmap
import sys


def advise_archive(mapping, page_policy):
    """Only this immutable historical mapping; never changes global THP settings."""
    if page_policy not in ('thp2m', 'base4k'):
        raise ValueError('unsupported historical archive page policy')
    advice = 'MADV_HUGEPAGE' if page_policy == 'thp2m' else 'MADV_NOHUGEPAGE'
    if page_policy == 'base4k' and (not hasattr(mapping, 'madvise') or not hasattr(mmap, advice)):
        raise RuntimeError('campaign base4k archive advice unavailable')
    if hasattr(mapping, 'madvise') and hasattr(mmap, advice):
        mapping.madvise(getattr(mmap, advice))


class RegisteredKVSlab:
    """A layer's newly committed K/V stored in a single contiguous mmap region."""

    def __init__(self, source_k, source_v, frame_count, frame_tokens, *, page_policy="base4k"):
        import torch
        if not sys.platform.startswith("linux") or not source_k.is_cuda:
            raise ValueError("THP_REGISTER_REQUIRES_LINUX_CUDA")
        if source_k.dtype != source_v.dtype or source_k.shape != source_v.shape:
            raise ValueError("THP_REGISTER_KV_SHAPE_DTYPE_MISMATCH")
        if source_k.ndim != 4 or source_k.shape[0] != 1:
            raise ValueError("THP_REGISTER_EXPECTS_B1_KV")
        if source_k.shape[1] != frame_count * frame_tokens:
            raise ValueError("THP_REGISTER_CURRENT_FRAME_GEOMETRY_MISMATCH")
        if source_k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("THP_REGISTER_KV_DTYPE_UNSUPPORTED")
        heads, dim = source_k.shape[2:]
        shape = (2, frame_count, frame_tokens, heads, dim)
        count = 2 * source_k.numel()
        nbytes = count * source_k.element_size()
        mapping = mmap.mmap(-1, nbytes,
                            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
        advise_archive(mapping, page_policy)
        # Fault/touch backing pages before CUDA registration. This is inside
        # the generation/commit wall-time, not a hidden pre-generation charge.
        storage = torch.frombuffer(mapping, dtype=source_k.dtype, count=count).reshape(shape)
        storage.zero_()
        ptr = storage.data_ptr()
        status = torch.cuda.cudart().cudaHostRegister(int(ptr), int(nbytes), 0)
        if status != torch.cuda.cudart().cudaError.success:
            raise RuntimeError(f"DRAFTKV_CUDA_HOST_REGISTER_FAILED: {status}")
        if not storage.is_pinned():
            raise RuntimeError("DRAFTKV_REGISTERED_TH P_NOT_PINNED".replace(" ", ""))
        self._mapping = mapping
        self._storage = storage
        self.bytes = nbytes
        self.frame_count = frame_count
        self.page_policy = page_policy

    def frame(self, rank):
        if not 0 <= rank < self.frame_count:
            raise IndexError(rank)
        return self._storage[0, rank], self._storage[1, rank]

    # Intentionally no eager cudaHostUnregister. Hook owns this object until
    # worker exit; host registration must outlive any outstanding B01 DMA.
    # CUDA tears down registered mappings when the worker process terminates.


class RegisteredArchiveArena:
    """Permanent layer/frame slices, prepared once inside measured execution."""
    def __init__(self,layers,frames,frame_tokens,heads=12,*,page_policy='base4k'):
        import torch
        import time
        begin=time.perf_counter()
        count=layers*frames*frame_tokens*heads*256
        self.bytes=count*2
        self._mapping=mmap.mmap(-1,self.bytes,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,
                                prot=mmap.PROT_READ|mmap.PROT_WRITE)
        advise_archive(self._mapping,page_policy)
        self.page_policy=page_policy
        self.storage=torch.frombuffer(self._mapping,dtype=torch.bfloat16,count=count).view(layers,frames,-1)
        self.storage.zero_()
        touched=time.perf_counter()
        status=torch.cuda.cudart().cudaHostRegister(self.storage.data_ptr(),self.bytes,0)
        if status!=torch.cuda.cudart().cudaError.success or not self.storage.is_pinned():
            raise RuntimeError(f'archive arena registration failed: {status}')
        self.touch_ms=(touched-begin)*1000
        self.register_ms=(time.perf_counter()-touched)*1000
