"""Compile the detected 5090 SP8 Wan transport signatures without a model forward.

This child-only preparation never launches the sampler or a Triton math kernel.
Actual module initialization and all transport work stay inside measured runtime.
"""
import hashlib
from pathlib import Path


def wan_specializations(plan, *, capability, device_name):
    from .catalog import MODELS
    from .sequence_parallel import local_heads
    from draftkv.streaming.device_profiles import RTX5090_SP8
    from draftkv.streaming.table2_native import runtime_config, transfer_units
    if (plan.get('algorithm') != 'draftkv' or plan.get('sp') != 8
            or plan.get('backbone') not in ('LingBot1.3B', 'LingBot14B')):
        return None
    cfg=(plan.get('algorithm_plugin') or {}).get('config', {})
    mode=cfg.get('retrieval_ablation', 'full')
    if (mode not in ('full', 'draft-only') or cfg.get('gpu_prepack', False)
            or cfg.get('gpu_device_route', False) or not cfg.get('gpu_krope_8x16', True)
            or not cfg.get('device_archive_plan', True)):
        return None
    spec=MODELS[plan['backbone']]
    heads=local_heads(spec.heads, spec.family, 8)
    runtime=runtime_config(cfg,capability=capability,device_name=device_name,
        physical_sp=8,heads=heads,default_slots=True)
    if not runtime or runtime.get('device_profile') != RTX5090_SP8.name:
        return None
    if runtime.get('head_group') != heads or runtime.get('submission') != 'wave':
        return None
    request=plan['request'];allocation=plan['effective_budget']
    if allocation['local']:
        return None
    h,w=request['output_height']//16,request['output_width']//16
    ft=h*w;regions=((h+7)//8)*((w+15)//16)
    from draftkv.algorithms.draftkv.block_policy import configured_protection
    from draftkv.algorithms.draftkv.experiment import experiment_scope
    with experiment_scope(cfg,plan['backbone']):
        protect=configured_protection(plan['backbone'])
    top=cfg.get('camera_top_chunks', 3)
    dma=set();rotations=set()
    for start in range(0,request['latent_frames'],spec.chunk):
        sink=min(allocation['sink'],start)
        capacity=min(allocation['retrieved'],max(0,start-sink))
        if not capacity:continue
        if mode=='full':
            p=min(protect,capacity,start-sink)
            first=(sink+spec.chunk-1)//spec.chunk
            stop=max(first,start//spec.chunk-2)
            candidates=min(top,stop-first)*spec.chunk+p
            budget=min(capacity,candidates)
        else:
            candidates=start-sink;budget=capacity
        if not budget:continue
        total=(sink+budget+spec.chunk)*ft;a=sink*ft;n=budget*ft
        splits=min(runtime['splits'],total)
        r=1 << max(0,(candidates*regions-1).bit_length())
        dma.add((r,splits,total,a,n,heads,runtime['reverse']))
        for unit in transfer_units(1,splits,total,a,n,reverse=runtime['reverse']):
            lo,hi=max(0,unit.source_begin),min(n,unit.source_end)
            if hi>lo:rotations.add(hi-lo)
    return dict(heads=heads,rotations=sorted(rotations),dma=sorted(dma),
        profile=RTX5090_SP8.name,frame_tokens=ft)


def prepare(plan, *, capability, device_name):
    shapes=wan_specializations(plan,capability=capability,device_name=device_name)
    if shapes is None:return None
    import torch
    from triton.runtime.jit import MockTensor
    from draftkv.streaming import prepacked, archive_producer
    bf16,i32,i64,f64=[MockTensor(t) for t in (torch.bfloat16,torch.int32,torch.int64,torch.float64)]
    for n in shapes['rotations']:
        prepacked._rotate_fetched(bf16,i64,i32,f64,n,shapes['heads'],
            preload=True,heads=shapes['heads'])
    counts,emit=archive_producer.kernels()
    for r,s,total,a,n,g,reverse in shapes['dma']:
        u=s
        compiled=counts.warmup(i64,i32,r,s,total,a,n,r,reverse,grid=(u,))
        _=compiled.run
        compiled=emit.warmup(i64,i32,i64,0x100000000,r,s,total,a,n,g,u,
            r,1 << (u-1).bit_length(),reverse,0,0,grid=(u,))
        _=compiled.run
    return dict(status='READY',profile=shapes['profile'],
        rotation_signatures=shapes['rotations'],dma_signature_count=len(shapes['dma']),
        numerical_kernel_launches=0,model_forwards=0,
        sources={str(Path(m.__file__).name):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                 for m in (prepacked,archive_producer)})
