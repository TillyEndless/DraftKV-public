from pathlib import Path
import subprocess,json
from draftkv.runtime.io import read,sha
from draftkv.streaming.integrity import verify_provider
from draftkv.algorithms.draftkv.config import validate_deployment
from draftkv.evaluators.interfaces import video_contract,verify_adapter,exact_git,specs
from draftkv.backbones.native import source_environment
from draftkv.streaming.dependency import verify_b01
from draftkv.streaming.native_sources import verify_external_native

def check(c,p,probe_hardware=False):
    issues=[];source=None;hardware={'status':'RUNTIME_NOT_TESTED'}
    def guarded(kind,call):
        try:return call()
        except (ValueError,OSError,KeyError,TypeError) as error:issues.append({'status':kind,'reason':str(error)});return None
    a=guarded('BLOCKED_MISSING_ASSET_CONFIG',lambda:read(p['asset_config'])) if p['asset_config'] else None
    if a is None:
        if not issues:issues.append({'status':'BLOCKED_MISSING_ASSET_CONFIG','reason':'Supply --asset-config'})
    elif p['mode']=='run':
        guarded('EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED',lambda:verify_b01(a))
        source=guarded('EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED',lambda:verify_provider(c,a['source_root']))
        if c['id']!='LongLive1.3B':guarded('EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED',lambda:verify_external_native(a['source_root']))
        if not a.get('cases'):issues.append({'status':'BLOCKED_MISSING_ASSET','reason':'No original case requests supplied'})
        for case in a.get('cases',[]):
            guarded('BLOCKED_MISSING_ASSET',lambda case=case:validate_deployment(c,read(case['config'])) if sha(case['config'])==case['sha256'] else (_ for _ in ()).throw(ValueError('Case config hash differs')))
        if not Path(a.get('model_python','')).is_file():issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':'Model interpreter missing'})
        checkpoint=Path(a.get('checkpoint',''))
        if not checkpoint.exists():issues.append({'status':'BLOCKED_MISSING_ASSET','reason':'Model checkpoint missing'})
        elif checkpoint.is_file() and a.get('checkpoint_sha256')!=sha(checkpoint):issues.append({'status':'BLOCKED_MISSING_ASSET','reason':'Model checkpoint hash differs'})
        elif checkpoint.is_dir() and not a.get('checkpoint_files'):issues.append({'status':'BLOCKED_MISSING_ASSET','reason':'Directory checkpoint requires checkpoint_files index'})
        for rel,wanted in a.get('checkpoint_files',{}).items():
            if guarded('BLOCKED_MISSING_ASSET',lambda rel=rel:sha(checkpoint/rel))!=wanted:issues.append({'status':'BLOCKED_MISSING_ASSET','reason':'Checkpoint shard hash differs: '+rel})
    else:
        guarded('PREFLIGHT_BLOCKED_MISSING_VIDEO',lambda:video_contract(c,p,a))
        evaluator=a.get('evaluators',{}).get(p['evaluator'])
        if evaluator:guarded('BLOCKED_MISSING_DEPENDENCY',lambda:verify_adapter(evaluator['adapter_root'],p['evaluator']))
        else:issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':'Missing pinned evaluator deployment'})
    if a and not p.get('generation_only',False):
        evaluator=a.get('evaluators',{}).get(p['evaluator'])
        if not evaluator:issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':'Missing evaluator deployment'})
        else:
            guarded('BLOCKED_MISSING_DEPENDENCY',lambda:verify_adapter(evaluator['adapter_root'],p['evaluator']))
            weights=guarded('BLOCKED_MISSING_DEPENDENCY',lambda:read(evaluator['weight_index']))
            if not weights:issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':'Evaluator weight index empty/missing'})
            for weight in weights or []:
                if guarded('BLOCKED_MISSING_DEPENDENCY',lambda weight=weight:sha(weight['path']))!=weight.get('sha256'):issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':'Evaluator weight hash differs'})
            if p['evaluator']!='vbench-long':guarded('BLOCKED_MISSING_DEPENDENCY',lambda:exact_git(evaluator['source'],specs()[p['evaluator']]['upstream_revision']))
            if not Path(evaluator.get('python','')).is_file():issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':'Evaluator interpreter missing'})
    if probe_hardware and a and a.get('model_python') and Path(a['model_python']).is_file():
        code="import json,torch;print(json.dumps({'cuda':torch.version.cuda,'torch':torch.__version__,'count':torch.cuda.device_count(),'devices':[{'name':torch.cuda.get_device_properties(i).name,'vram':torch.cuda.get_device_properties(i).total_memory,'sm':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]}))"
        try:
            result=subprocess.run([a['model_python'],'-c',code],env=source_environment(a.get('source_root','.'),p['physical_devices'],p['workspace'],model_python=a['model_python']),text=True,capture_output=True,timeout=45,check=True);hardware=json.loads(result.stdout)
            required=1 if p['mode']=='evaluate' else p['runtime_sp']
            if hardware['count']<required:issues.append({'status':'UNSUPPORTED_TOPOLOGY','reason':'Insufficient physical GPUs; SP is unchanged'})
            hardware['status']='DEVICE_INVENTORY_ONLY_NOT_DISTRIBUTED_KERNEL_ACCEPTANCE'
            if p['mode']=='run' and c['id']!='LongLive1.3B':
                import os,shutil
                env=source_environment(a.get('source_root','.'),p['physical_devices'],p['workspace'],model_python=a['model_python'])
                tools={name:shutil.which(name,path=env.get('PATH')) for name in ('ninja','nvcc','c++')}
                hardware['native_build_tools']=tools
                if not all(tools.values()):issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':'Native build executables missing from the selected model environment: '+json.dumps(tools)})
                include=env.get('CUTLASS_INCLUDE_DIR')
                if not include or not (Path(include)/'cute/tensor.hpp').is_file():issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':'Set CUTLASS_INCLUDE_DIR to the authorized B01 CUTLASS include directory before native compilation'})
        except (OSError,ValueError,subprocess.SubprocessError) as error:issues.append({'status':'BLOCKED_MISSING_DEPENDENCY','reason':str(error)})
    return {'status':'PREFLIGHT_STATIC_DEPENDENCIES_PASS' if not issues else 'PREFLIGHT_BLOCKED','source':source,'hardware':hardware,'issues':issues,'public_repository_readiness':'SEPARATE_REPOSITORY_AUDIT','runtime_tested':False}
