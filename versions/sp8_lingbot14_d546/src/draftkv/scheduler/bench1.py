"""Bench1-only group workflows on the existing SQLite queue and shared broker.

Planning never probes a GPU. Execution is explicit and uses the same broker,
process identity, immutable attempts and leases as the existing Scheduler. No
process is killed here. A surviving descendant retains its lease for recovery.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from draftkv.registry import Registry
from draftkv.runtime.config import Config, ConfigurationError, parse_devices, resolve_path, repo_root
from draftkv.runtime.state import RunState, write_json
from draftkv.runtime.engine import validated_artifacts
from .common import atomic_json, canonical, digest, matches, object_hash, process_identity, read_json
from .resources import Resources, ResourceBlocked
from .runner import owned_processes, classify, set_attempt
from .store import Queue, ACTIVE, BLOCKED, FAILURES


def plan(root, config_path, devices, policy, backbone=None, algorithm=None, case_ids=None, limit=None, *, benchmark="bench1"):
    config = Config(root, str(config_path))
    policy = normalized_policy(policy, root)
    pool = parse_devices(devices)
    result = []
    for group in config.registry.select(benchmark, backbone, algorithm):
        assignment = policy.get('assignments', {}).get('/'.join(group.key),
                      policy.get('assignments', {}).get(group.backbone, list(pool)))
        selected = parse_devices(','.join(map(str, assignment)))
        if not set(selected) <= set(pool):
            raise ConfigurationError('per-group devices must be a subset of the explicit physical pool')
        if benchmark == 'bench2' and selected != pool:
            raise ConfigurationError('Bench2 assignment must preserve the exact explicit physical device list and SP')
        resolved = config.resolve(group, selected)
        resolved['selection'] = dict(case_ids=case_ids, limit_cases=limit)
        dataset = resolved.get('dataset', {})
        result.append(dict(**group.to_dict(), case_id='ALL480' if benchmark == 'bench2' else 'ALL98', manifest_sha256=dataset.get('sha256'),
            runtime_config=dict(path=str(config.path), sha256=digest(config.path)),
            repo_root=str(root), python=sys.executable, devices=list(selected), sp=len(selected),
            input_mode=group.protocol.input_mode, selection=resolved['selection'],
            memory=policy.get('memory', {}).get('/'.join(group.key), policy.get('memory', {}).get(group.backbone, {})),
            selection_sha256=object_hash(resolved['selection']),
            policy_sha256=policy['_source_sha256'],
            workflow='generate_then_official_orbia' if benchmark == 'bench2' else 'generate_then_official_vbench_long'))
    return result


def cli_argv(config, command, *, source_run=None, resume_run=None, preflight=False):
    argv = [config['python'], '-m', 'draftkv.cli', command, '--benchmark', config.get('benchmark', 'bench1'),
        '--backbone', config['backbone'], '--algorithm', config['algorithm'],
        '--devices', ','.join(map(str, config['devices'])), '--repo-root', config['repo_root'],
        '--config', config['runtime_config']['path']]
    selection = config['selection']
    if selection.get('case_ids'):
        argv += ['--case-ids', ','.join(selection['case_ids'])]
    if selection.get('limit_cases'):
        argv += ['--limit-cases', str(selection['limit_cases'])]
    if source_run: argv += ['--source-run', source_run]
    if resume_run: argv += ['--resume', resume_run]
    if preflight: argv += ['--preflight']
    return argv


def locked_config(config):
    if digest(config['runtime_config']['path']) != config['runtime_config']['sha256']:
        raise ValueError('frozen runtime config changed; use a new queue workspace')


def generation_valid(config, run_id, *, complete=True):
    from draftkv.runtime.state import git_sha
    from draftkv.runtime.config import json_sha256
    from draftkv.runtime.engine import safe_name
    safe_name(run_id, 'run ID')
    directory = Path(config['repo_root'])/'runs'/config.get('benchmark', 'bench1')/config['backbone']/config['algorithm']/run_id
    if directory.is_symlink():
        raise ValueError('source run cannot be a symlink')
    state = RunState(**read_json(directory/'state.json'))
    requested = read_json(directory/'request_config.json')
    resolved = read_json(directory/'resolved_config.json')
    code_sha=json_sha256({str(p.relative_to(Path(config['repo_root']))):digest(p)
        for p in sorted((Path(config['repo_root'])/'src').rglob('*.py'))})
    if ((state.benchmark, state.backbone, state.algorithm) != (config.get('benchmark', 'bench1'),config['backbone'],config['algorithm'])
            or state.run_id != run_id or state.git_sha != git_sha(Path(config['repo_root']))
            or state.git_root != config['repo_root']
            or state.code_sha != code_sha
            or state.selected_devices != config['devices'] or state.sp != config['sp']
            or state.config_path != config['runtime_config']['path']
            or state.request_sha != json_sha256(requested) or state.config_sha != json_sha256(resolved)
            or state.dataset_sha != config['manifest_sha256']
            or requested.get('selection') != config['selection'] or not state.completed_case_ids):
        raise ValueError('generation source identity/revision changed')
    validated_artifacts(directory, state, complete=complete)
    if complete and state.stages.get('generate') != 'SUCCEEDED':
        raise ValueError('source generation is incomplete')
    return state


class Bench1Resources(Resources):
    def snapshot(self, config, **kwargs):
        snapshot = super().snapshot(config, **kwargs)
        selected = set(config['devices'])
        snapshot['gpus'] = [gpu for gpu in snapshot['gpus'] if gpu['index'] in selected]
        if len(snapshot['gpus']) != len(selected):
            raise ResourceBlocked('assigned physical GPU is absent')
        return snapshot


@contextmanager
def lock_queue(queue):
    with (queue.root/'bench1.dispatch.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def lock_task(policy, task_id):
    """The same frozen task in distinct workspaces must share one execution lock."""
    root = Path(policy['task_lock_root']); root.mkdir(parents=True, exist_ok=True)
    with (root/(task_id+'.lock')).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceBlocked('same task is active in another workspace') from exc
        stream.seek(0)
        payload = stream.read()
        if payload:
            try:
                previous = json.loads(payload)
                token = previous['attempt_token']
                if (previous.get('schema_version')!=1 or not isinstance(token, str)
                        or len(token)!=32 or any(c not in '0123456789abcdef' for c in token)):
                    raise ValueError('invalid attempt token')
            except (ValueError, KeyError, TypeError) as exc:
                raise ResourceBlocked('shared task lock identity is corrupt; manual reconciliation required') from exc
            if owned_processes(token):
                raise ResourceBlocked('same task has live descendants from another workspace')
        # Closing the parent's descriptor must not unlock a live CLI that
        # inherited the same open-file description through pass_fds.
        yield stream.fileno()


def bind_task_attempt(descriptor, token):
    if descriptor is None:
        return
    payload = (canonical({'schema_version': 1, 'attempt_token': token})+'\n').encode()
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    os.write(descriptor, payload)
    os.fsync(descriptor)


def normalized_policy(policy, root):
    # Freeze resolved targets, not ${ENV} spelling: changing the environment
    # must not silently switch the shared task lock or lease authority.
    result = {k:v for k,v in policy.items() if k!='_source_sha256'}
    for field in ('legacy_leases','broker_python','task_lock_root','broker_state'):
        if field in result:
            result[field] = str(resolve_path(result[field],root))
    if 'legacy_queues' in result:
        result['legacy_queues'] = [str(resolve_path(p,root)) for p in result['legacy_queues']]
    if 'broker' in result:
        result['broker'] = dict(result['broker'],path=str(resolve_path(result['broker']['path'],root)))
    result['_source_sha256']=object_hash(result)
    return result


def preflight(queue, task):
    config = json.loads(task['config']); locked_config(config)
    process = subprocess.run(cli_argv(config,'run',preflight=True), cwd=config['repo_root'],
        capture_output=True,text=True,env={**os.environ,'PYTHONPATH':str(Path(config['repo_root'])/'src')})
    try:
        report = json.loads(process.stdout)
    except (ValueError, TypeError):
        report = {'status':'INVALID_PREFLIGHT_OUTPUT','stdout':process.stdout,'stderr':process.stderr,
                  'exit_code':process.returncode}
    ok = process.returncode == 0 and all(row['status']=='READY' for row in report.get('runs',[])) and len(report.get('runs',[]))==1
    with queue.transaction():
        queue.transition(task['id'],'PREFLIGHT')
        queue.audit(task['id'],'BENCH1_NATIVE_PREFLIGHT',report)
        queue.audit(task['id'],'BENCH1_PREFLIGHT_PROCESS',dict(stderr=process.stderr,exit_code=process.returncode))
        queue.transition(task['id'],'READY' if ok else 'BLOCKED_ENVIRONMENT', '' if ok else 'native CLI preflight rejected; see immutable run logs')
    return ok


def _run_cli(queue, task, token, config, command, folder, resource, *, task_lock_fd=None, **options):
    argv = cli_argv(config,command,**options)
    atomic_json(folder/(command+'.argv.json'),argv)
    atomic_json(folder/(command+'.launch.json'), {'attempt_token': token, 'phase': command,
                                                'argv': argv, 'time': time.time()})
    environment = dict(os.environ, DRAFTKV_ATTEMPT_TOKEN=token, DRAFTKV_ATTEMPT_PHASE=command,
        PYTHONPATH=str(Path(config['repo_root'])/'src'))
    with (folder/(command+'.stdout.log')).open('x') as out, (folder/(command+'.stderr.log')).open('x') as err:
        process = subprocess.Popen(argv,cwd=config['repo_root'],env=environment,stdout=out,stderr=err,
                                   stdin=subprocess.DEVNULL,start_new_session=True,
                                   pass_fds=(() if task_lock_fd is None else (task_lock_fd,)))
    identity = process_identity(process.pid)
    with queue.transaction():
        set_attempt(queue,token,child=canonical(identity) if identity else None)
        queue.audit(task['id'],'BENCH1_CLI_STARTED',dict(argv=argv,identity=identity,phase=command))
    renewed = time.monotonic()
    while process.poll() is None:
        if time.monotonic()-renewed >= 60:
            resource.renew(token); renewed=time.monotonic()
        time.sleep(.25)
    output = read_json(folder/(command+'.stdout.log'))
    rows = output.get('runs',[])
    if len(rows)!=1:
        raise RuntimeError('CLI did not return one isolated run')
    row = rows[0]
    atomic_json(folder/(command+'.result.json'),row)
    if row.get('stages',{}).get('generate') == 'SUCCEEDED' or command=='run' and row.get('completed_case_ids'):
        saved = json.loads(queue.task(task['id'])['result'] or '{}')
        saved['generation_run_id']=row['run_id']
        saved['generation_complete']=row.get('stages',{}).get('generate')=='SUCCEEDED'
        queue.db.execute('UPDATE tasks SET result=? WHERE id=?',(canonical(saved),task['id']))
    if process.returncode or row['status']!='SUCCEEDED':
        # Classify the actual child failure while retaining all original logs.
        source = Path(config['repo_root'])/'runs'/config.get('benchmark', 'bench1')/config['backbone']/config['algorithm']/row['run_id']
        # Model subprocess errors live inside the sequential per-case directory.
        # The outer exception alone only says that torchrun exited nonzero.
        case_logs = sorted(source.glob('cases/*/stderr.log'))[-1:]
        diagnostics = '\n'.join(p.read_text(errors='replace')[-8000:]
                                 for p in (folder/(command+'.stderr.log'), source/'stderr.log', *case_logs) if p.is_file())
        raise RuntimeError('native CLI '+command+' failed; run_id='+row['run_id']+'\n'+diagnostics)
    return row


def execute_one(queue, policy, task):
    config=json.loads(task['config'])
    policy=normalized_policy(policy,Path(config['repo_root']))
    if config['policy_sha256']!=policy['_source_sha256']:
        raise ValueError('frozen batch policy changed; use a new queue workspace')
    with lock_task(policy, task['id']) as descriptor:
        return _execute_locked(queue, policy, task, task_lock_fd=descriptor)


def _execute_locked(queue, policy, task, *, task_lock_fd=None):
    if policy.get('scope') != json.loads(task['config']).get('benchmark', 'bench1'):
        raise ValueError('explicit benchmark shared-broker policy must match the task')
    config=json.loads(task['config']); locked_config(config)
    if config['policy_sha256'] != policy.get('_source_sha256', object_hash(policy)):
        raise ValueError('frozen batch policy changed; use a new queue workspace')
    if not preflight(queue,task): return None
    token=queue.claim(task['id']); resource=Bench1Resources(queue,policy)
    held=False
    try:
        with queue.transaction(): set_attempt(queue,token,worker=canonical(process_identity(os.getpid())))
        bind_task_attempt(task_lock_fd, token)
        folder=queue.root/'attempts'/task['id']/token; folder.mkdir(parents=True,exist_ok=False)
        atomic_json(folder/'config.json',config); atomic_json(folder/'policy.json',policy)
        resource.acquire(config,task['id'],token); held=True
        with queue.transaction(): queue.transition(task['id'],'RUNNING',token=token)
        saved=json.loads(queue.task(task['id'])['result'] or '{}')
        generation=saved.get('generation_run_id')
        with queue.transaction(): queue.transition(task['id'],'GENERATING',token=token)
        if generation and saved.get('generation_complete'):
            generation_valid(config,generation)
        else:
            if generation: generation_valid(config,generation,complete=False)
            result=_run_cli(queue,task,token,config,'run',folder,resource,resume_run=generation,
                            task_lock_fd=task_lock_fd)
            generation=result['run_id']; generation_valid(config,generation)
        with queue.transaction(): queue.transition(task['id'],'EVALUATING',token=token)
        result=_run_cli(queue,task,token,config,'evaluate',folder,resource,source_run=generation,
                        task_lock_fd=task_lock_fd)
        saved=json.loads(queue.task(task['id'])['result'] or '{}'); saved['evaluation_run_id']=result['run_id']
        with queue.transaction():
            queue.db.execute('UPDATE tasks SET result=? WHERE id=?',(canonical(saved),task['id']))
            queue.transition(task['id'],'COMPLETE',token=token)
            set_attempt(queue,token,ended=time.time(),exit_code=0)
    except (Exception,KeyboardInterrupt) as exc:
        import traceback
        if 'folder' in locals():
            (folder/'scheduler.traceback.log').write_text(traceback.format_exc())
        if queue.task(task['id'])['state'] in ACTIVE|{'READY'}:
            previous = queue.task(task['id'])['state']
            state,_=classify(str(exc))
            if previous == 'READY':
                state = 'BLOCKED_GPU' if isinstance(exc, ResourceBlocked) else 'BLOCKED_ENVIRONMENT'
            elif previous == 'EVALUATING' and state not in FAILURES:
                state = 'FAILED_RUNTIME'
            with queue.transaction():
                queue.transition(task['id'],state,str(exc),token=token)
                set_attempt(queue,token,ended=time.time(),exit_code=1,diagnostic=str(exc))
        raise
    finally:
        # Never release or steal a lease while this attempt has live descendants.
        if held and not owned_processes(token): resource.release(token)
    return task['id']


def recover_generation(queue, task, config, token, folder):
    """Find only this attempt's run; never infer success from directory age."""
    saved=json.loads(task['result'] or '{}')
    if saved.get('generation_run_id'):
        return
    parent=Path(config['repo_root'])/'runs'/config.get('benchmark', 'bench1')/config['backbone']/config['algorithm']
    candidates=[]
    for path in parent.glob('*/scheduler_attempt.json'):
        try:
            binding=read_json(path)
        except (OSError,ValueError):
            continue
        if binding.get('attempt_token')==token and binding.get('phase')=='run':
            candidates.append((path,binding))
    if len(candidates)>1:
        raise ResourceBlocked('ambiguous generation runs for this attempt; refusing regeneration')
    output=folder/'run.stdout.log'
    hint=None
    if output.is_file():
        try:
            rows=read_json(output).get('runs',[])
            if len(rows)==1: hint=rows[0]['run_id']
        except (OSError,ValueError,KeyError,TypeError):
            pass
    binding=None
    if candidates:
        path,binding=candidates[0]
        if path.is_symlink() or path.parent.is_symlink():
            raise ResourceBlocked('scheduler attempt metadata cannot be a symlink')
        run_id=path.parent.name
        if hint is not None and hint!=run_id:
            raise ResourceBlocked('stdout disagrees with persistent attempt identity')
    elif hint:
        # Compatibility with earlier immutable attempts that predate metadata.
        run_id=hint
    elif output.exists() or (folder/'run.launch.json').exists():
        raise ResourceBlocked('generation was launched but its run identity cannot be verified; refusing regeneration')
    else:
        return  # Claimed before launch: no model could have been invoked.
    state=generation_valid(config,run_id,complete=False)
    if binding:
        expected={'schema_version':1,'attempt_token':token,'phase':'run','run_id':run_id,
            'benchmark':config.get('benchmark', 'bench1'),'backbone':config['backbone'],'algorithm':config['algorithm'],
            'git_sha':state.git_sha,'request_sha':state.request_sha,
            'config_path':config['runtime_config']['path'],
            'config_file_sha256':config['runtime_config']['sha256']}
        if binding!=expected:
            raise ResourceBlocked('persistent attempt identity differs from the validated run snapshots')
        if state.status not in ('SUCCEEDED','FAILED','BLOCKED_INTEGRATION'):
            # Liveness was proved absent under the global lock. Retain the
            # original state before sealing an interrupted run for native resume.
            directory=parent/run_id
            atomic_json(folder/'generation.recovered_state.json',{
                'attempt_token':token,'state_sha256':digest(directory/'state.json'),
                'original_state':asdict(state),'reason':'all owned processes exited'})
            state.status='FAILED'; state.error_type='RecoveredExitedSchedulerAttempt'
            state.exit_code=1; state.finished_at=datetime.now(timezone.utc).isoformat()
            if state.phase and state.stages.get(state.phase)=='RUNNING':
                state.stages[state.phase]='FAILED'
            write_json(directory/'state.json',asdict(state))
    saved.update(generation_run_id=state.run_id,
                 generation_complete=state.stages.get('generate')=='SUCCEEDED')
    with queue.transaction():
        queue.db.execute('UPDATE tasks SET result=? WHERE id=?',(canonical(saved),task['id']))
        queue.audit(task['id'],'BENCH1_GENERATION_RECOVERED',{
            'run_id':state.run_id,'attempt_token':token,'persistent_identity':binding is not None})


def resume(queue, policy):
    reports=[]
    for task in queue.tasks():
        if task['state'] in BLOCKED and not task['attempt']:
            config=json.loads(task['config']); locked_config(config)
            active_policy=normalized_policy(policy,Path(config['repo_root']))
            if config['policy_sha256'] != active_policy['_source_sha256']:
                raise ValueError('frozen batch policy changed')
            with lock_task(active_policy,task['id']), queue.transaction():
                if queue.db.execute('SELECT 1 FROM leases WHERE task_id=?',(task['id'],)).fetchone():
                    raise ResourceBlocked('unclaimed task still has a lease; manual reconciliation required')
                queue.db.execute('UPDATE tasks SET state=?,reason=? WHERE id=?',
                                 ('PENDING','EXPLICIT_RESUME_AFTER_PREFLIGHT',task['id']))
                queue.audit(task['id'],'PREFLIGHT_RESUME',{'previous_state':task['state']})
            reports.append(dict(task_id=task['id'],status='PENDING')); continue
        if task['state'] not in ACTIVE|FAILURES|BLOCKED|{'READY'} or not task['attempt']: continue
        token=task['attempt']
        attempt=queue.db.execute('SELECT * FROM attempts WHERE token=?',(token,)).fetchone()
        alive=owned_processes(token) or (attempt and any(matches(json.loads(attempt[k])) for k in ('worker','child') if attempt[k]))
        if alive:
            reports.append(dict(task_id=task['id'],status='LIVE_ATTEMPT_RETAINED')); continue
        config=json.loads(task['config']); locked_config(config)
        active_policy=normalized_policy(policy,Path(config['repo_root']))
        if config['policy_sha256'] != active_policy['_source_sha256']:
            raise ValueError('frozen batch policy changed')
        try:
            with lock_task(active_policy, task['id']):
                resource=Bench1Resources(queue,active_policy)
                resource.recover_owned(task['id'],token)
                resource.release(token)
                recover_generation(queue,task,config,token,queue.root/'attempts'/task['id']/token)
        except (OSError,ValueError,KeyError,TypeError,ResourceBlocked) as exc:
            state=queue.task(task['id'])['state']
            target='STALLED' if state in ACTIVE else ('BLOCKED_ENVIRONMENT' if state=='READY' else state)
            with queue.transaction():
                queue.transition(task['id'],target,'RECOVERY_REQUIRES_REVIEW: '+str(exc),token=token)
                queue.audit(task['id'],'BENCH1_RECOVERY_BLOCKED',{'reason':str(exc)})
            reports.append(dict(task_id=task['id'],status='BLOCKED_RECOVERY',reason=str(exc)))
            continue
        if task['state'] in ACTIVE:
            with queue.transaction(): queue.transition(task['id'],'STALLED','owned worker exited; no process termination performed')
        # Explicit --resume authorizes a new immutable attempt, preserving source run IDs.
        with queue.transaction():
            queue.db.execute('UPDATE tasks SET state=?,attempt=NULL,reason=? WHERE id=?',('PENDING','EXPLICIT_BENCH1_RESUME',task['id']))
            queue.audit(task['id'],'BENCH1_RESUME',{'previous_attempt':token,'result_retained':queue.task(task['id'])['result']})
        reports.append(dict(task_id=task['id'],status='PENDING'))
    return reports


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    actions=parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','dry-run','preflight','run-once','status','resume'):
        actions.add_argument('--'+action,action='store_true')
    parser.add_argument('--config'); parser.add_argument('--policy')
    parser.add_argument('--devices',default='0'); parser.add_argument('--backbone'); parser.add_argument('--algorithm')
    parser.add_argument('--case-ids'); parser.add_argument('--limit-cases',type=int)
    parser.add_argument('--repo-root')
    args=parser.parse_args(argv)
    try:
        root=repo_root(args.repo_root)
        policy=read_json(resolve_path(args.policy,root)) if args.policy else {}
        if args.prepare or args.dry_run:
            config_path=args.config or os.environ.get('DRAFTKV_CONFIG')
            if not config_path: raise ValueError('explicit --config or DRAFTKV_CONFIG required')
            ids=args.case_ids.split(',') if args.case_ids else None
            if ids and (len(set(ids))!=len(ids) or any(not cid or cid.strip()!=cid for cid in ids)): raise ValueError('unique case IDs required')
            if args.limit_cases is not None and args.limit_cases<1: raise ValueError('positive case limit required')
            plans=plan(root,resolve_path(config_path,root),args.devices,policy,args.backbone,args.algorithm,ids,args.limit_cases)
            if args.dry_run:
                print(json.dumps(dict(status='PLANNED',groups=plans,group_count=len(plans),gpu_executed=False),indent=2)); return 0
            queue=Queue(resolve_path(args.workspace,root))
            result={'submitted':[queue.submit(config) for config in plans]}
        else:
            queue=Queue(resolve_path(args.workspace,root))
            with lock_queue(queue):
                if args.status: result={'tasks':queue.tasks()}
                elif args.resume:
                    if policy.get('scope')!='bench1': raise ValueError('resume requires explicit Bench1 policy')
                    result={'recovered':resume(queue,normalized_policy(policy,root))}
                elif args.preflight:
                    result={'preflight':[dict(task_id=t['id'],ready=preflight(queue,t)) for t in queue.tasks() if t['state'] not in ACTIVE|FAILURES|{'COMPLETE'} and not t['attempt']]}
                else:
                    policy=normalized_policy(policy,root)
                    candidates=[t for t in queue.tasks() if t['state'] not in ACTIVE|FAILURES|{'COMPLETE'} and not t['attempt']]
                    result={'started':execute_one(queue,policy,candidates[0]) if candidates else None}
        print(json.dumps(result,indent=2)); return 0
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
        print(json.dumps(dict(status='BLOCKED',reason=str(exc)))); return 2

if __name__=='__main__': raise SystemExit(main())
