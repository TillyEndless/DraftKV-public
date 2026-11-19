from pathlib import Path
import argparse,time
from draftkv.runtime.io import read,write,require
from draftkv.algorithms.draftkv.config import config,binding,binding_hash
from draftkv.backbones.native import prepare,generate
from draftkv.evaluators.interfaces import evaluate,aggregate

def execute(phase,path):
    p=read(path);c=config(p['backbone']);require(p['binding']==binding(c) and p['binding_sha256']==binding_hash(c),'Canonical binding changed')
    require(p['algorithm']=='draftkv' and phase in [s['phase'] for s in p['launch_stages']],'Phase is not allowed')
    require(p['runtime_sp']==c['runtime_topologies'][p['topology']],'SP topology changed')
    a=read(p['asset_config']);out=Path(p['workspace']);receipt={'phase':phase,'status':'FAILED','backbone':c['id'],'paper_numerical_reproduction':False};start=time.time()
    try:
        if phase=='prepare':prepare(c,p,a)
        elif phase=='generate':generate(c,p,a)
        elif phase=='evaluate':evaluate(c,p,a)
        elif phase=='aggregate':aggregate(c,p)
        else:raise ValueError('Unknown phase')
        receipt['status']='COMPLETE'
    except Exception as error:receipt['error']=str(error);raise
    finally:receipt['elapsed_s']=time.time()-start;write(out/(phase+'-receipt.json'),receipt)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['prepare','generate','evaluate','aggregate']);p.add_argument('--plan',required=True);a=p.parse_args(argv);execute(a.phase,a.plan)
if __name__=='__main__':main()
