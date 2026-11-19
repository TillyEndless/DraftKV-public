"""Standalone adapter subprocess, imported under the selected original source tree."""
import hashlib,json,os,sys
from pathlib import Path

def run(payload):
    d=json.load(open(payload));c=d['config'];p=d['plan'];a=d['assets'];videos=d['videos'];e=a['evaluators'][p['evaluator']];out=Path(d['output'])
    if p['evaluator']=='orbia':
        from draftkv.benchmarks.orbia import ORBIADatasetResolver,ORBIASpec
        from draftkv.evaluators.orbia import ORBIAOfficialEvaluator,ORBIAEvaluatorSpec,ORBIAVideos
        from draftkv.evaluators.orbia_video import build_video_manifest
        lock=e['dataset_lock'];assert hashlib.sha256(Path(lock).read_bytes()).hexdigest()==e['dataset_lock_sha256']
        assert hashlib.sha256(Path(e['config']).read_bytes()).hexdigest()==e['config_sha256']
        ds=ORBIADatasetResolver(Path(e['adapter_root']),lock=lock).resolve(ORBIASpec(**a['dataset']))
        directory=Path(p['workspace'])/'orbia-videos';directory.mkdir(exist_ok=False)
        for row in videos['cases']:(directory/(row['case_id']+'.mp4')).symlink_to(Path(row['video']).resolve())
        manifest=Path(p['workspace'])/'orbia-video-manifest.json'
        metadata=build_video_manifest(ds,directory,manifest,model=c['id']+'_draftkv',interface=e['interface'],run_config=e['run_config'],checkpoint=e['checkpoint'],conditions=e['conditions'],case_ids=[r['case_id'] for r in videos['cases']])
        assert set(e.get('options',{})) <= {'threads','geometry_root','installation_receipt','installation_receipt_sha256'}
        spec=ORBIAEvaluatorSpec(official_root=e['source'],python=e['python'],config=e['config'],devices=','.join(str(i) for i in range(len(p['physical_devices']))),**e.get('options',{}))
        report=ORBIAOfficialEvaluator(spec,lock=lock).evaluate(ds,ORBIAVideos(manifest,metadata['sha256'],directory),out)
        assert report.as_dict()['status']=='COMPLETE'
    else:
        from draftkv.evaluators.bench1 import Bench1Evaluator
        rows=videos['cases'];generation={'status':'SUCCEEDED','run_id':c['paper_row'],'dataset_sha256':a['dataset_sha256'],'weight_hashes':a['weight_hashes'],'case_ids':[r['case_id'] for r in rows],'artifacts':[{'case_id':r['case_id'],'path':r['video'],'sha256':r['sha256']} for r in rows]}
        result=Bench1Evaluator(Path(e['adapter_root']),e['config']).evaluate(a['dataset'],generation,c['id'],out,metadata={'method':'draftkv'})
        assert result['status']=='COMPLETE';out.mkdir(parents=True,exist_ok=True);(out/'bridge-result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':run(sys.argv[1])
