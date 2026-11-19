"""CPU checks of preservation, CLI dispatch and the existing measurement contract."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
BACKBONES=['LongLive1.3B','LingBot1.3B','LingBot14B','SolarWM33B']


def run(args,env=None):
    selected={**os.environ,**(env or {}),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
    guard=os.environ.get('DRAFTKV_MINIMAL_TEST_GUARD')
    if guard:selected['PYTHONPATH']=guard+os.pathsep+selected.get('PYTHONPATH','')
    return subprocess.run([sys.executable,*args],cwd=ROOT,env=selected,capture_output=True,text=True)


@pytest.mark.parametrize('bb',BACKBONES)
def test_static_plan_preserves_defaults_and_sp(bb):
    p=run(['-m','draftkv_minimal.cli','plan','--backbone',bb])
    assert p.returncode==0,p.stderr
    d=json.loads(p.stdout)
    assert d['profile_mode']=='off'
    assert d['runtime_sp']==(1 if bb=='LongLive1.3B' else 8)
    assert d['algorithm']=='draftkv'


def test_all_original_source_bodies_are_unchanged():
    rows=json.loads((ROOT/'release_keep_manifest.json').read_text())['records']
    for r in rows:
        f=ROOT/r['path']
        assert hashlib.sha256(f.read_bytes()).hexdigest()==r['release_sha256'],r['path']
        if f.suffix in ('.py','.cu','.cpp','.h'):
            assert r['source_sha256']==r['release_sha256'],r['path']


@pytest.mark.parametrize('bb',BACKBONES)
def test_off_avoids_diagnostics_and_preserves_required_events(bb):
    source=ROOT/'src/draftkv_minimal/verified'/bb/'public/src'
    code='''
from types import SimpleNamespace
from pathlib import Path
import tempfile
from draftkv.runtime import profiling as pm
pm.configure('off')
def forbidden(*a,**k):raise AssertionError('OFF touched diagnostic')
torch=SimpleNamespace(cuda=SimpleNamespace(Event=forbidden,max_memory_allocated=forbidden))
pm.time.perf_counter_ns=forbidden
pm.time.perf_counter=forbidden
assert pm.event(torch) is None
assert pm.clock_ns()==0 and pm.peak_allocated(torch,0) is None
event=SimpleNamespace(synchronize=forbidden,elapsed_time=forbidden)
pm.diagnostic_sync(event)
assert pm.elapsed(event,event) is None
assert pm.relative({'origin':event,'join':event})=={}
assert pm.run_profiled({},lambda:17)==17
with tempfile.TemporaryDirectory() as t:
 assert pm.aggregate({'workspace':t},[]) is None
 assert not list(Path(t).iterdir())
seen=[]
torch.cuda.Event=lambda **kw:seen.append(kw) or object()
assert pm.event(torch,required=True) is not None
assert seen==[{'enable_timing':False}]
pm.configure('summary');pm.event(torch)
assert seen[-1]=={'enable_timing':True}
'''
    p=run(['-c',code],{'PYTHONPATH':str(source)})
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('bb',BACKBONES)
def test_summary_aggregates_fixture_observations_without_historical_constants(bb):
    source=ROOT/'src/draftkv_minimal/verified'/bb/'public/src'
    code='''
import tempfile
from pathlib import Path
from draftkv.algorithms.draftkv.config import config
from draftkv.runtime.plan import build
from draftkv.runtime.io import write
from draftkv.runtime import profiling as pm
pm.configure('summary')
with tempfile.TemporaryDirectory() as t:
 p=build(config(BACKBONE),t,profile_mode='summary')
 for rank in range(p['runtime_sp']):
  totals=pm.Totals()
  totals.observe(route='cpu-fixture',fetch_attention_ms=10+rank,h2d_transfer_ms=8,h2d_bytes=123,receive_buffer_bytes=2**20,source='CPU contract fixture')
  write(Path(t)/'profile/ranks'/('rank'+str(rank)+'.json'),{'rank':rank,'cases':[{'case_id':'fixture','status':'COMPLETE','generation_e2e_s':2+rank,'peak_allocated_bytes':(rank+1)*2**30,'peak_reserved_bytes':(rank+2)*2**30,'routes':totals.routes}]})
 result=pm.aggregate(p,[{'case_id':'fixture','generated_frames':100,'output_video_duration_s':10}])
 assert result['generation_e2e_s']==1+p['runtime_sp']
 assert result['fetch_attention_ms']==9+p['runtime_sp']
 assert result['h2d_transfer_ms']==8
 assert result['h2d_bytes']==123
 assert result['receive_buffer_mib']==1
 assert len(result['per_rank'])==p['runtime_sp']
 assert result['streaming_overlap']['status']=='UNAVAILABLE'
'''.replace('BACKBONE',repr(bb))
    p=run(['-c',code],{'PYTHONPATH':str(source)})
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('mode',['trace','nsys','detailed'])
def test_only_off_summary_are_public(mode):
    p=run(['-m','draftkv_minimal.cli','plan','--backbone','LingBot14B','--profile',mode])
    assert p.returncode!=0


@pytest.mark.parametrize('bb',BACKBONES)
def test_original_native_worker_and_planner_import_without_other_baselines(bb):
    source=ROOT/'src/draftkv_minimal/verified'/bb/'provider/src'
    p=run(['-c','import draftkv.backbones.worker; import draftkv.backbones.planning'],{'PYTHONPATH':str(source)})
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('algorithm',['native','worldkv','camera_only','draft_only'])
def test_nonrelease_algorithms_are_rejected(algorithm):
    p=run(['-m','draftkv_minimal.cli','plan','--backbone','LingBot14B','--algorithm',algorithm])
    assert p.returncode!=0
    assert 'invalid choice' in p.stderr


@pytest.mark.parametrize('bb',BACKBONES)
def test_missing_external_dependencies_fail_closed(bb):
    source=ROOT/'src/draftkv_minimal/verified'/bb/'public/src'
    code='''
import tempfile
from pathlib import Path
from draftkv.streaming.dependency import verify_b01
from draftkv.streaming.integrity import verify_provider
from draftkv.algorithms.draftkv.config import config
for operation in ('b01','provider'):
 with tempfile.TemporaryDirectory() as t:
  try:
   verify_b01({}) if operation=='b01' else verify_provider(config(BACKBONE),Path(t))
  except (ValueError,RuntimeError,FileNotFoundError,AssertionError):pass
  else:raise AssertionError('missing dependency silently accepted')
'''.replace('BACKBONE',repr(bb))
    p=run(['-c',code],{'PYTHONPATH':str(source)})
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('bb,values',[
    ('LongLive1.3B',(3,0,6,3,2,None)),
    ('LingBot1.3B',(3,0,11,4,3,3)),
    ('LingBot14B',(3,0,11,4,3,3)),
    ('SolarWM33B',(0,0,25,5,5,7)),
])
def test_scientific_configuration_identity(bb,values):
    p=run(['-m','draftkv_minimal.cli','plan','--backbone',bb]);assert p.returncode==0,p.stderr
    d=json.loads(p.stdout)
    c=d.get('configuration',d.get('config',{}))
    # Read the original catalog via the original CLI, not duplicated configs.
    shown=run(['-m','draftkv_minimal.cli','show','--backbone',bb]);assert shown.returncode==0,shown.stderr
    c=json.loads(shown.stdout);b=c['budget'];m=c['method']
    assert (b['sink'],b['local'],b['retrieved'],b['current'],m['protect_recent_frames'],m.get('camera_top_chunks'))==values
