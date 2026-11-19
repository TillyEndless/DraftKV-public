# SPDX-License-Identifier: Apache-2.0
"""SUMMARY-only preparation of the same public native modules used by generation.

The exact external latency admission observer remains source-verified. Compilation
is performed before model construction, outside the measured generation interval.
No sampler, attention fallback, random inputs or model warmup is executed here.
"""
from importlib import import_module
from pathlib import Path
import os
import time
from draftkv.runtime import profiling
from draftkv.runtime.io import require,sha,write


def prepare(plan, trace=None):
    if not profiling.enabled():return None
    if '_latency_build_preparation' in plan:return plan['_latency_build_preparation']
    import torch
    from draftkv.backbones.latency_build import build_backends, POLICY, CACHE_VARIABLES
    rank=int(os.environ.get('RANK','0'));device=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(device)
    backends=build_backends(plan)
    if not backends and torch.cuda.get_device_capability() in ((8,9),(9,0),(12,0)):
        backends=['device_route']
    started=time.perf_counter();extensions=[]
    for backend in backends:
        module=import_module('draftkv.streaming.'+backend)
        require(module.__name__.startswith('draftkv.algorithms.draftkv.historical.orbia.streaming.'),'Build admission cannot load an unextracted algorithm module')
        native=module.native()
        extensions.append({'name':native.__name__,'module_file':str(Path(native.__file__).resolve()),'sha256':sha(native.__file__),'source_identity':native.source_identity})
    receipt={'status':'READY','policy':POLICY,'phase':'environment_preparation','excluded_from_e2e':True,'cache_bindings':{name:os.environ.get(name) for name in CACHE_VARIABLES},'backends':backends,'extensions':extensions,'preparation_wall_s':time.perf_counter()-started,'torch':str(torch.__version__),'cuda':torch.version.cuda}
    plan['_latency_build_preparation']=receipt
    from draftkv.runtime.io import read
    release=read(os.environ['DRAFTKV_PUBLIC_RELEASE_PLAN'])
    write(Path(release['workspace'])/'profile/preparation'/('rank'+str(rank)+'.json'),receipt)
    return receipt
