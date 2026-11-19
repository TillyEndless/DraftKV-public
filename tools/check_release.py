"""CPU-only source/distribution integrity checks; no model or CUDA imports."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN=('.so','.a','.pt','.pth','.safetensors','.mp4','.pyc')


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',type=Path)
    parser.add_argument('--sdist',type=Path)
    parser.add_argument('--rebuilt-wheel',type=Path)
    args=parser.parse_args()
    rows=json.loads((ROOT/'release_keep_manifest.json').read_text())['records']
    body=[]
    for row in rows:
        rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        data=(ROOT/rel).read_bytes()
        assert sha(data)==row['release_sha256'],str(rel)
        if rel.suffix in ('.py','.cu','.cpp','.h'):
            assert row['source_sha256']==row['release_sha256'],str(rel)
            body.append(row)
        if rel.suffix=='.py':ast.parse(data,filename=str(rel))
    assert len(body)==846 and len({r['release_sha256'] for r in body})==285
    ext=json.loads((ROOT/'verified_runtime_source_map.json').read_text())['external_dependencies']
    assert all(not (ROOT/r['path']).exists() for r in ext),'Restricted dependency unexpectedly bundled'
    wheel_payload=None
    if args.wheel:
        with zipfile.ZipFile(args.wheel) as z:
            names=z.namelist()
            assert not any('/.git/' in n or '/.release-private/' in n or '/receipts/' in n or n.endswith(FORBIDDEN) for n in names)
            for row in rows:assert sha(z.read(row['path'].removeprefix('src/')))==row['release_sha256'],row['path']
            wheel_payload={n:sha(z.read(n)) for n in names if n.startswith('draftkv_minimal/')}
    if args.sdist:
        with tarfile.open(args.sdist) as tar:
            members=tar.getmembers()
            for m in members:
                assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                assert not m.issym() and not m.islnk()
                assert '/.release-private/' not in m.name and '/.git/' not in m.name
                assert not m.name.endswith(FORBIDDEN)
            names={m.name.split('/',1)[1] for m in members if m.isfile() and '/' in m.name}
            for rel in ['CITATION.cff','docs/INSTALL.md','docs/PROFILING.md','docs/THIRD_PARTY_NOTICES.md','configs/assets.example.json']:
                assert rel in names,rel
    if args.rebuilt_wheel:
        assert wheel_payload is not None
        with zipfile.ZipFile(args.rebuilt_wheel) as z:
            rebuilt={n:sha(z.read(n)) for n in z.namelist() if n.startswith('draftkv_minimal/')}
        assert rebuilt==wheel_payload,'sdist rebuild changed runtime payload'
    print(json.dumps({'status':'PASS','source_records':len(rows),'unchanged_implementation_bodies':len(body),'distinct_body_hashes':285,'external_dependency_records':len(ext),'wheel_checked':bool(args.wheel),'sdist_checked':bool(args.sdist),'sdist_runtime_equal':bool(args.rebuilt_wheel)}))


if __name__=='__main__':main()
