"""Separate build compatibility layer. Original sources and setup files are immutable.

This entry point compiles ONLY when explicitly executed in a future CUDA build gate.
"""

import json
import os
from pathlib import Path
import sys

# Support direct execution with the selected model Python, without installing the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def extension_options(name, sources, includes, library_dirs, spec):
    """Torch extension factories mutate their input lists in place.

    Each module owns its arguments; toolchain receipt identity must retain the
    checked input rather than acquiring Torch search paths during construction.
    """
    cuda = any(path.endswith('.cu') for path in sources)
    return dict(name=name, sources=list(sources), include_dirs=list(includes),
                library_dirs=list(library_dirs),
                extra_compile_args={"cxx": list(spec['cxx_flags']), "nvcc": list(spec['nvcc_flags'])}
                if cuda else list(spec['cxx_flags']))


def main():
    from draftkv.vendor.b01.build import (MODULES, CXX_FLAGS, NVCC_FLAGS, FA, assert_abi,
                                          current_abi, check_toolchain, check_clean_checkout, dry_run)
    from draftkv.vendor.b01.cache import cache_key, finalize_cache, load_native
    from draftkv.vendor.b01.integrity import B01Error, extract_sources
    spec_path = Path(os.environ["B01_BUILD_SPEC"])
    spec = json.loads(spec_path.read_text())
    cache_root = Path(os.environ["B01_CACHE_ROOT"]).resolve()
    expected = dry_run(spec["arch"], spec["abi"], spec["clean_commit"], spec["cuda_home"], cache_root, spec_path)["spec"]
    if spec != expected:
        raise B01Error("CUDA_ABI", "stale or modified build specification")
    import torch
    if torch.cuda.is_initialized():
        raise B01Error("LogicError", "CPU native build must not initialize CUDA")
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
    assert_abi(spec["abi"], current_abi(torch))
    if os.environ.get("CUDA_HOME") != spec["cuda_home"] or os.environ.get("TORCH_CUDA_ARCH_LIST") != spec["arch"]:
        raise B01Error("CUDA_ABI", "build environment differs from specification")
    if sys.argv[1:] != ["build_ext"]:
        raise B01Error("LogicError", "only explicit build_ext supported; no install or flag overrides")
    check_clean_checkout(spec["clean_commit"])
    toolchain = check_toolchain(spec)
    checked_key = cache_key(spec, toolchain)
    build_dir = cache_root / checked_key
    build_dir.mkdir(parents=True, exist_ok=True)
    if any(build_dir.iterdir()):
        if (build_dir / "receipt.json").is_file():
            load_native(build_dir, spec)
            print(f"Verified existing native build: {build_dir}")
            return
        raise B01Error("CUDA_ABI", "cache directory already populated; validate and reuse or quarantine it")
    source = extract_sources(build_dir / "source")
    includes = toolchain["includes"] + [str(source / FA), str(source / FA / "src"), str(source / "third_party/cutlass/include")]
    modules = []
    for name, sources in MODULES.items():
        cuda = any(s.endswith(".cu") for s in sources)
        options = extension_options(name, [str(source / p) for p in sources], includes,
                                    toolchain['library_dirs'], spec)
        if not cuda:
            # Original CppExtension omitted CUDA Torch libraries. Link them
            # explicitly in this compatibility layer; do not edit the snapshot.
            options["libraries"] = ["c10_cuda", "torch_cuda", "cudart", "cuda"]
        modules.append((CUDAExtension if cuda else CppExtension)(**options))
    setup(name="draftkv-original-b01", version="0", ext_modules=modules,
          script_args=["build_ext", "--build-lib", str(build_dir), "--build-temp", str(build_dir / "objects")],
          cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)})
    if cache_key(spec, toolchain) != checked_key:
        raise B01Error("CUDA_ABI", "extension construction mutated the checked build identity")
    finalize_cache(build_dir, spec, toolchain)
    if torch.cuda.is_initialized():
        raise B01Error("LogicError", "native export verification initialized CUDA")


if __name__ == "__main__":
    main()
