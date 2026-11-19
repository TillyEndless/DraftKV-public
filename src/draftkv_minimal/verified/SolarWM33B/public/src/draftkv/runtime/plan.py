from pathlib import Path
import sys
from draftkv.algorithms.draftkv.config import binding,binding_hash
from draftkv.runtime.io import require
from draftkv.algorithms.draftkv.core import verify_core

def devices_for(c,topology,devices,evaluation=False):
    require(topology in c['runtime_topologies'],'Unverified topology; historical SP is never rewritten')
    sp=c['runtime_topologies'][topology]
    values=list(range(1 if evaluation else sp)) if devices is None else [int(v) for v in devices.split(',')]
    require(values and len(values)==len(set(values)) and min(values)>=0,'Devices must be unique nonnegative indices')
    require(evaluation or len(values)==sp,'UNSUPPORTED_TOPOLOGY: physical device count must equal SP')
    return sp,values

def build(c,workspace,assets=None,devices=None,topology='historical',evaluation=False,evaluator=None,video_map=None,profile_mode='off',generation_only=False):
    require(profile_mode in ('off','summary'),'Profile mode must be off or summary')
    sp,physical=devices_for(c,topology,devices,evaluation)
    selected=evaluator or c['evaluation']['quality']
    require(selected in (c['evaluation']['quality'],'revisit'),'Evaluator does not match backbone')
    require(selected!='revisit' or c['evaluation']['revisit'],'Revisit unsupported')
    require(evaluation or selected!='revisit','Revisit is evaluation-only')
    root=str(Path(workspace).resolve());stages=[]
    require(not (evaluation and generation_only),'Generation-only is incompatible with evaluation')
    phases=['evaluate','aggregate'] if evaluation else (['prepare','generate'] if generation_only else ['prepare','generate','evaluate','aggregate'])
    for phase in phases:
        argv=[sys.executable,'-m','draftkv.runtime.executor','--phase',phase,'--plan',str(Path(root)/'execution-plan.json')]
        stages.append({'phase':phase,'argv':argv})
    return {'schema_version':1,'generation_only':generation_only,'profile_mode':profile_mode,'backbone':c['id'],'algorithm':'draftkv','paper_row':c['paper_row'],'binding':binding(c),'binding_sha256':binding_hash(c),'public_core':verify_core(c['id']),'mode':'evaluate' if evaluation else 'run','historical_sp':c['historical_sp'],'runtime_sp':sp,'topology':topology,'physical_devices':physical,'workspace':root,'asset_config':str(Path(assets).resolve()) if assets else None,'video_map':str(Path(video_map).resolve()) if video_map else None,'evaluator':selected,'launch_stages':stages,'distributed_worker_argv':['MODEL_PYTHON','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node='+str(sp),'-m','draftkv.runtime.core_worker','--release-plan',str(Path(root)/'execution-plan.json'),'--plan','PREPARED_CASE_PLAN'],'static_plan_status':'STATIC_PLAN_PASS','runtime':'RUNTIME_NOT_TESTED','public_control_source_status':'PACKAGED','source_materialization':'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED','gpu_runtime_status':'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED'}
