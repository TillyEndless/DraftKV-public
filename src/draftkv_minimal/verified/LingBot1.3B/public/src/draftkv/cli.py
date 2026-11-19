"""The only public interface: four canonical DraftKV Full configurations."""
import argparse,json,sys
from pathlib import Path
from draftkv.algorithms.draftkv.config import BACKBONES,config,select
from draftkv.runtime.io import ReleaseError,require,read,write
from draftkv.runtime.plan import build
from draftkv.runtime.preflight import check

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['list','show','plan','preflight','run','evaluate','status','static-check','dependencies']);parser.add_argument('--algorithm',choices=['draftkv'],default='draftkv');parser.add_argument('--backbone',choices=BACKBONES);parser.add_argument('--table',choices=['table1','table2','table4']);parser.add_argument('--devices');parser.add_argument('--topology',choices=['historical','validated-sp1'],default='historical');parser.add_argument('--workspace');parser.add_argument('--asset-config');parser.add_argument('--video-map');parser.add_argument('--evaluator',choices=['vbench-long','orbia','revisit']);parser.add_argument('--generation-only',action='store_true');parser.add_argument('--probe-hardware',action='store_true');parser.add_argument('--profile',choices=['off','summary'],default='off');args=parser.parse_args(argv)
    try:
        if args.operation=='list':print(json.dumps([config(b) for b in BACKBONES],indent=2));return 0
        require(args.backbone is not None,'--backbone is required');c=select(args.algorithm,args.backbone,args.table)
        if args.operation=='dependencies':
            from draftkv.streaming.dependency import requirement
            from draftkv.streaming.native_sources import native_contract
            print(json.dumps({**requirement(), 'native_source_contract':native_contract()},indent=2));return 0
        if args.operation=='show':print(json.dumps({**c,'profile_mode':args.profile},indent=2));return 0
        require(not(args.table=='table4' and args.operation=='run'),'Table4 is evaluation-only')
        workspace=args.workspace or str(Path('outputs')/c['id'])
        if args.operation=='status':
            p=Path(workspace);print(json.dumps({f.name:read(f) for f in p.glob('*-receipt.json')},indent=2));return 0
        evaluation=args.operation=='evaluate' or args.table=='table4' or args.evaluator=='revisit'
        require(args.operation!='run' or not evaluation,'Revisit is evaluation-only')
        plan=build(c,workspace,args.asset_config,args.devices,args.topology,evaluation,args.evaluator or ('revisit' if args.table=='table4' else None),args.video_map,profile_mode=args.profile,generation_only=args.generation_only)
        if args.operation in ('plan','static-check'):
            from draftkv.streaming.integrity import source_lock
            lock=source_lock(c['source_profile']);require(lock['revision']==c['runtime_source_revision'] and lock['files'],'Missing static source binding')
            if args.operation=='static-check':print(json.dumps({'status':'STATIC_CLI_READY','profile_mode':args.profile,'backbone':c['id'],'binding_sha256':plan['binding_sha256'],'external_runtime_dependency':'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED','historical_sp':c['historical_sp'],'runtime_sp':plan['runtime_sp'],'launch_stages':plan['launch_stages']},indent=2))
            else:print(json.dumps(plan,indent=2))
            return 0
        report=check(c,plan,args.probe_hardware or args.operation in ('run','evaluate'))
        if args.operation=='preflight':print(json.dumps(report,indent=2));return 0 if not report['issues'] else 2
        require(not report['issues'],'Preflight blocked: '+json.dumps(report['issues']))
        require(not (Path(workspace)/'execution-plan.json').exists(),'Workspace already contains a plan; choose a fresh workspace')
        write(Path(workspace)/'execution-plan.json',plan)
        from draftkv.runtime.executor import execute
        for stage in plan['launch_stages']:execute(stage['phase'],Path(workspace)/'execution-plan.json')
        return 0
    except (ReleaseError,OSError,KeyError,TypeError) as error:print(json.dumps({'status':'BLOCKED','error':str(error)}),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
