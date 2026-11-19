from importlib.resources import files
import json
from draftkv.runtime.io import require,fingerprint
BACKBONES=('LongLive1.3B','LingBot1.3B','LingBot14B','SolarWM33B')
def config(backbone):
    require(backbone in BACKBONES,'Unsupported backbone')
    return json.loads(files('draftkv').joinpath('data','configs',backbone+'.json').read_text())
def select(algorithm,backbone,table=None):
    require(algorithm=='draftkv','Only DraftKV Full is supported')
    c=config(backbone)
    require(table in (None,c['table'],'table4'),'Paper selector does not match backbone')
    require(table!='table4' or c['evaluation']['revisit'],'Revisit evaluation is unavailable for LongLive')
    return c
def binding(c):
    return {k:c[k] for k in ('id','algorithm','runtime_source_revision','budget','method','historical_sp')}
def binding_hash(c):return fingerprint(binding(c))
def validate_deployment(c,deployment):
    actual=deployment.get('effective_budget',{})
    require(all(actual.get(k)==v for k,v in c['budget'].items()),'Budget differs from canonical Full configuration')
    inert={'block_size':None,'keep_ratio':None,'retrieved_equivalent':False}
    require(all(k in c['budget'] or k in inert and v==inert[k] for k,v in actual.items()),'Unexpected budget field')
    plugin=deployment.get('algorithm_plugin',{});method=dict(plugin.get('config',{}))
    require(method.pop('retrieval_ablation','full')=='full','Only Full retrieval is permitted')
    hardware={'b01','kernel_binding','kernel_dir','kernel_abi','host_pages_binding','host_pages_binding_sha256','device_profile','build_dir','hardware_profile'}
    require(not set(deployment)&{'ablation','experiment_override','generation_override'},'Experiment override forbidden')
    require(plugin.get('entry')=='src/draftkv/algorithms/draftkv/entry.py','Canonical Full plugin required')
    require(all(method.get(k)==v for k,v in c['method'].items()),'Method settings differ from canonical Full configuration')
    require(set(method)<=set(c['method'])|hardware,'Unrecognized method setting')
    require(not method.get('gpu_prepack') and not method.get('gpu_device_route'),'Alternate retrieval execution is not selected')
    return deployment
