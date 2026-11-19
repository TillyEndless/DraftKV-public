from importlib.resources import files
from pathlib import Path
import json,subprocess
from draftkv.runtime.io import read,sha,require,write
from draftkv.backbones.native import source_environment

def specs():return json.loads(files('draftkv').joinpath('data','evaluators.json').read_text())
def verify_adapter(root,kind):
    profile='longlive' if kind=='vbench-long' else 'evaluation'
    lock=json.loads(files('draftkv').joinpath('data','locks',profile+'.json').read_text())
    for rel,wanted in lock['files'].items():
        if profile=='longlive' and not rel.startswith('src/draftkv/evaluators/'):continue
        require((Path(root)/rel).is_file() and sha(Path(root)/rel)==wanted,'Evaluator adapter source missing/changed: '+rel)
    return lock['revision']

def video_contract(c,p,a):
    path=p.get('video_map') or a.get('video_map') or str(Path(p['workspace'])/'video-map.json');doc=read(path)
    require(doc.get('algorithm')=='draftkv' and doc.get('backbone')==c['id'],'Video belongs to a different algorithm/backbone')
    require(doc.get('binding_sha256')==p['binding_sha256'] and doc.get('source_revision')==c['runtime_source_revision'],'Video generation source/config differs')
    rows=doc.get('cases',[]);require(rows and len({r['case_id'] for r in rows})==len(rows),'Missing/duplicate video cases')
    for row in rows:require(Path(row['video']).suffix.lower()=='.mp4' and sha(row['video'])==row['sha256'],'Video missing/changed')
    return doc

def evaluate(c,p,a):
    require(p['evaluator'] in (c['evaluation']['quality'],'revisit'),'Wrong evaluator')
    v=video_contract(c,p,a);e=a['evaluators'][p['evaluator']];root=Path(p['workspace']);out=root/p['evaluator']
    verify_adapter(e['adapter_root'],p['evaluator'])
    weights=read(e['weight_index']);require(weights,'No local evaluator weight index')
    for weight in weights:require(sha(weight['path'])==weight['sha256'],'Evaluator weight missing/changed')
    env=source_environment(e['adapter_root'],p['physical_devices'],root)
    if p['evaluator']=='revisit':
        require(c['evaluation']['revisit'],'Revisit unsupported');exact_git(e['source'],specs()['revisit']['upstream_revision'])
        video_map=root/'revisit-input.json';write(video_map,{c['id']+'/draftkv':{r['case_id']:r['video'] for r in v['cases']}})
        env['WBENCH_OFFLINE_ASSETS']='1'
        cmd=[e['python'],str(Path(e['adapter_root'])/'scripts/table3_reprojection_eval.py'),'run','--video-map',str(video_map),'--backbone',c['id'],'--algorithm','draftkv','--expected-cases',str(len(v['cases'])),'--wbench-root',e['source'],'--wbench-python',e['python'],'--wbench-weights',e['weights'],'--gpu','0','--hash-videos','--allow-insufficient-support','--output',str(out)]
    else:
        if p['evaluator']=='orbia':exact_git(e['source'],specs()['orbia']['upstream_revision'])
        payload=root/'evaluation-input.json';write(payload,{'config':c,'plan':p,'assets':a,'videos':v,'output':str(out)})
        cmd=[e['python'],str(files('draftkv').joinpath('evaluators','bridge.py')),str(payload)]
    subprocess.run(cmd,cwd=e['adapter_root'],env=env,check=True)
    write(root/'evaluation-receipt.json',{'evaluator':p['evaluator'],'output':str(out),'source':e.get('source'),'binding_sha256':p['binding_sha256'],'paper_numerical_reproduction':False})

def exact_git(root,revision):
    got=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();require(got==revision,'Evaluator upstream revision differs')
    require(not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True).strip(),'Evaluator checkout is not clean')

def aggregate(c,p):
    root=Path(p['workspace']);receipt=read(root/'evaluation-receipt.json');out=Path(receipt['output']);kind=receipt['evaluator']
    name={'orbia':'short_video_scores.json','revisit':'table3_report.json','vbench-long':'bridge-result.json'}[kind];path=out/name;data=read(path)
    if kind=='orbia':
        require(len(data['rows'])==1,'Ambiguous quality aggregate');row=data['rows'][0]
        metrics={k:row[k] for k in ['T1','T2','T3','T4','T5','T6','Overall']}
    elif kind=='revisit':metrics=data['group_results'][c['id']+'/draftkv']['report']['metrics']
    else:metrics={**data['scores'],'quality_score':data.get('quality_score')}
    write(root/'result.json',{'backbone':c['id'],'algorithm':'draftkv','binding_sha256':p['binding_sha256'],'evaluator':kind,'metrics':metrics,'result_source_sha256':sha(path),'paper_numerical_reproduction':False})
