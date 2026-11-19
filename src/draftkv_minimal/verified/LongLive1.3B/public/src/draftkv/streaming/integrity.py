"""Hash validation of independently supplied source; no restricted code is bundled."""
from importlib.resources import files
from pathlib import Path
import json
from draftkv.runtime.io import require,sha

def source_lock(profile):
    require(profile in ('longlive','lingbot','solar','evaluation'),'Unknown source profile')
    return json.loads(files('draftkv').joinpath('data','locks',profile+'.json').read_text())
def verify_provider(c,root):
    root=Path(root).resolve();lock=source_lock(c['source_profile'])
    require(lock['revision']==c['runtime_source_revision'],'Source revision binding changed')
    require(root.is_dir(),'BLOCKED_MISSING_SOURCE_PROVIDER')
    for rel,wanted in lock['files'].items():
        p=root/rel
        require(p.is_file(),'BLOCKED_MISSING_SOURCE: '+rel)
        require(sha(p)==wanted,'SOURCE_HASH_MISMATCH: '+rel)
    return {'status':'EXACT_EXTERNAL_SOURCE_BYTES_VERIFIED','file_count':len(lock['files']),'revision':lock['revision'],'git_commit_object_asserted':False}
