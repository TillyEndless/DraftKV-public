from pathlib import Path
import hashlib,json,os
class ReleaseError(ValueError):pass
def require(condition,message):
    if not condition:raise ReleaseError(message)
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
