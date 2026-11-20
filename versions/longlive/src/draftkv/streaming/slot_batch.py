"""Slots-only unit-batched DMA; immutable Dense executor is untouched.

The ABI comes from CUDA's public driver header. Older driver/toolkit installs
retain the original producer. PyTorch's old CUPTI may omit batch transfers;
use a modern Nsight trace with copyCount for activity and bandwidth reporting.
"""
from functools import lru_cache
from pathlib import Path
import ctypes,hashlib,os,re

@lru_cache(None)
def batch_include():
    from torch.utils.cpp_extension import CUDA_HOME
    roots=[Path(CUDA_HOME)] if CUDA_HOME else []
    roots.extend(Path('/usr/local').glob('cuda-*'))
    candidates=[]
    for root in roots:
        for include in (root/'include',root/'targets/x86_64-linux/include'):
            header=include/'cuda.h'
            if not header.is_file():continue
            match=re.search(r'^#define CUDA_VERSION\s+(\d+)',header.read_text(),re.M)
            if match and int(match[1])>=12080:candidates.append((int(match[1]),str(include)))
    return min(candidates)[1] if candidates else None

def available(device=None):
    import torch
    if torch.cuda.get_device_capability(device)!=(9,0):return False
    version=ctypes.c_int()
    rc=ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(version))
    return rc==0 and version.value>=12080 and batch_include() is not None

def make_producer(device):
    if not available(device):return None
    return native().AsyncProducer(True)

@lru_cache(None)
def native():
    import torch
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('slot_batch_cuda')/'binding.cpp'
    include=batch_include()
    if include is None:raise RuntimeError('slot batch producer requires CUDA 12.8+ driver headers')
    identity=hashlib.sha256(source.read_bytes()+f'{torch.__version__}:{torch.version.cuda}:{include}'.encode()).hexdigest()[:16]
    root=Path(os.environ.get('DRAFTKV_PREPACK_BUILD','/tmp/draftkv-slot-batch'))/('slot-batch-'+identity)
    root.mkdir(parents=True,exist_ok=True)
    module=load(name='draftkv_slot_batch_'+identity,sources=[str(source)],
        extra_include_paths=[include],
        extra_cflags=['-O3','-std=c++17'],extra_ldflags=['-lcuda'],
        with_cuda=True,build_directory=str(root),verbose=False)
    module.source_identity=identity
    return module
