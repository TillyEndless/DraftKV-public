"""Invoke selected original workers without changing their attention implementations."""
from pathlib import Path
import os,subprocess,sys
from draftkv.algorithms.draftkv.config import validate_deployment
from draftkv.runtime.io import read,write,sha,require
from draftkv.streaming.integrity import verify_provider

def source_environment(source,devices,workspace,*,public_core=False):
    cache=Path(workspace)/'cache';cache.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PYTHONPATH':str(Path(source)/'src'),'CUDA_VISIBLE_DEVICES':','.join(map(str,devices)),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
    if public_core:env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env['PYTHONPATH']
    for key,name in [('TMPDIR','tmp'),('DRAFTKV_PREPACK_BUILD','device-route'),('TORCH_EXTENSIONS_DIR','torch'),('TRITON_CACHE_DIR','triton')]:
        p=cache/name;p.mkdir(exist_ok=True);env[key]=str(p)
    return env

def prepare(c,p,a):
    root=Path(p['workspace']);verify_provider(c,a['source_root']);prepared=[]
    for case in a['cases']:
        cid=case['case_id'];require(cid and all(ch.isalnum() or ch in '_-' for ch in cid),'Unsafe case id')
        require(sha(case['config'])==case['sha256'],'Deployment config hash mismatch');cfg=validate_deployment(c,read(case['config']))
        require(Path(cfg['python']).resolve()==Path(a['model_python']).resolve(),'Model Python differs')
        require(Path(cfg['algorithm_plugin']['root']).resolve()==Path(a['source_root']).resolve(),'Plugin source differs')
        lockname='solarwm_source_lock.json' if c['id']=='SolarWM33B' else 'source_locks.json'
        require(sha(cfg['source_lock'])==sha(Path(a['source_root'])/'configs/backbones'/lockname),'Native source lock differs')
        out=root/'cases'/cid;require(not out.exists(),'Case output already exists')
        payload={'source':a['source_root'],'benchmark':'bench1' if c['table']=='table1' else 'bench2','backbone':c['id'],'sp':p['runtime_sp'],'config':cfg,'checkpoint':a['checkpoint'],'output':str(out)}
        path=root/(cid+'-prepare.json');write(path,payload)
        code="""import json,sys
from pathlib import Path
from draftkv.backbones.planning import make_plan
from draftkv.registry import Group
from draftkv.backbones.catalog import MODELS
from draftkv.backbones.budgets import effective_budget
d=json.load(open(sys.argv[1]));g=Group(d['benchmark'],d['backbone'],'draftkv');b=effective_budget(MODELS[g.backbone],'draftkv').to_dict();assert all(b.get(k)==v for k,v in d['config']['effective_budget'].items());d['config']['effective_budget']=b
w=make_plan(Path(d['source']),g,d['sp'],d['config'],d['checkpoint'],Path(d['output']));Path(d['output']).mkdir(parents=True,exist_ok=True);(Path(d['output'])/'backbone_plan.json').write_text(json.dumps(w,indent=2))"""
        subprocess.run([a['model_python'],'-c',code,str(path)],env=source_environment(a['source_root'],[],root),cwd=a['source_root'],check=True)
        path=out/'backbone_plan.json';worker=read(path)
        require(worker['sp']==p['runtime_sp'],'Prepared SP drift')
        require(all(worker['effective_budget'].get(k)==v for k,v in c['budget'].items()),'Prepared budget drift')
        require(worker['algorithm']=='draftkv','Prepared algorithm drift')
        prepared.append({'case_id':cid,'plan':str(path),'plan_sha256':sha(path)})
    write(root/'prepared-cases.json',prepared)

def resume_decode(p,a,w):
    root=Path(w['run_dir']);failure=read(root/'traces/worker_failure_rank0.json')
    require(failure.get('message')=='KV_CONSTRAINT: sampling tensors remain before independent VAE decode','Unrelated generation failure; no resume')
    sampling=read(root/'sampling_receipt.json');latent=root/'complete_latents.safetensors'
    require(sampling.get('status')=='COMPLETE' and sampling.get('request')==w['request'],'Sampling not complete')
    require(sampling.get('source_commit')==w['repo_commit'] and sha(latent)==sampling['latents_sha256'],'Latent source/hash mismatch')
    receipt={'original_sampler_exit_code':1,'original_guard_retained':True,'sampler_rerun':False,'latent_sha256':sampling['latents_sha256']};write(root/'decode-resume.json',receipt)
    subprocess.run([a['model_python'],'-m','draftkv.runtime.core_worker','--release-plan',str(Path(p['workspace'])/'execution-plan.json'),'--plan',str(root/'backbone_plan.json'),'--decode-latents',str(latent)],cwd=a['source_root'],env=source_environment(a['source_root'],p['physical_devices'][:1],p['workspace'],public_core=True),check=True)
    d=read(root/'decode_receipt.json');require(d['status']=='COMPLETE' and d['latents_sha256']==sampling['latents_sha256'] and d['video_sha256']==sha(w['output']),'Decode receipt mismatch')
    receipt['decoder_exit_code']=0;write(root/'decode-resume.json',receipt)

def generate(c,p,a):
    require(p['mode']=='run','Evaluation cannot generate');verify_provider(c,a['source_root']);rows=[]
    for case in read(Path(p['workspace'])/'prepared-cases.json'):
        require(sha(case['plan'])==case['plan_sha256'],'Prepared plan changed');w=read(case['plan'])
        require(w['sp']==p['runtime_sp'] and w['algorithm']=='draftkv','Worker configuration drift')
        cmd=[a['model_python'],'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node='+str(p['runtime_sp']),'-m','draftkv.runtime.core_worker','--release-plan',str(Path(p['workspace'])/'execution-plan.json'),'--plan',case['plan']]
        try:subprocess.run(cmd,cwd=a['source_root'],env=source_environment(a['source_root'],p['physical_devices'],p['workspace'],public_core=True),check=True)
        except subprocess.CalledProcessError:
            if c['id']!='SolarWM33B':raise
            resume_decode(p,a,w)
        video=Path(w['output']);require(video.is_file() and video.stat().st_size>0,'Missing generated MP4')
        import json
        probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,avg_frame_rate,width,height,codec_name,duration:format=duration,size','-of','json',str(video)],capture_output=True,text=True,check=True)
        metadata=json.loads(probe.stdout);stream=metadata['streams'][0]
        row={'case_id':case['case_id'],'video':str(video),'sha256':sha(video),'generated_frames':int(stream['nb_read_frames']),'output_video_duration_s':float(metadata['format']['duration']),'video_metadata':metadata}
        rows.append(row)
    from draftkv.runtime.profiling import aggregate
    aggregate(p,rows)
    write(Path(p['workspace'])/'video-map.json',{'backbone':c['id'],'algorithm':'draftkv','binding_sha256':p['binding_sha256'],'source_revision':c['runtime_source_revision'],'runtime_sp':p['runtime_sp'],'cases':rows})
