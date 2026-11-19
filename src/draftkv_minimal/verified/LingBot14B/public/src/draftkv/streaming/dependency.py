from importlib.resources import files
from pathlib import Path
import json
from draftkv.runtime.io import require,sha,read

def requirement():return json.loads(files('draftkv').joinpath('data','b01.json').read_text())
def verify_b01(assets):
    r=requirement();b=assets.get('b01')
    require(b is not None,'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED: configure b01.source_archive, manifest, cache_directory and build_spec; see docs/INSTALL.md')
    require(sha(b['source_archive'])==r['source_archive_sha256'],'B01 archive SHA256 differs; no substitution permitted')
    require(sha(b['manifest'])==r['original_manifest_sha256'],'B01 manifest SHA256 differs')
    require(Path(b['cache_directory']).is_dir(),'B01 compiled cache missing')
    require(sha(b['build_spec'])==b['build_spec_sha256'],'B01 build-spec SHA256 differs')
    spec=read(b['build_spec']);require(spec.get('abi') and spec.get('source_generation'),'B01 build specification lacks original ABI/source generation')
    return {'status':'EXTERNAL_RUNTIME_DEPENDENCY_FILES_VERIFIED','kernel_execution_verified':False}
