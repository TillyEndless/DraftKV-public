"""CPU-safe plan generation; explicit toolchain checks for a later build gate."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig

from .integrity import ROOT, B01Error, canonical_hash, digest, source_generation

ARCHES = {"8.9": "sm_89", "9.0": "sm_90", "10.0": "sm_100", "12.0": "sm_120"}
CXX_FLAGS = ["-O3", "-std=c++17"]
NVCC_FLAGS = CXX_FLAGS + [
    "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-DFLASHATTENTION_DISABLE_DROPOUT", "-DFLASHATTENTION_DISABLE_ALIBI",
    "-DFLASHATTENTION_DISABLE_SOFTCAP", "-DFLASHATTENTION_DISABLE_LOCAL",
    "-lineinfo", "--ptxas-options=-v",
]
FA = "fa2_custom/flash-attention-v2.8.3-streaming/csrc/flash_attn"
MODULES = {
    "producer_plan_ext": ["build/producer_plan.cpp"],
    "frontier_streamwrite_ext": ["build/frontier_streamwrite_ext.cpp"],
    "fa2_streaming_consumer_ext": ["fa2_custom/streaming_binding.cpp", "fa2_custom/streaming_producer.cu", FA + "/src/flash_fwd_hdim128_bf16_sm80.cu"],
    "b01_ready_ext": ["fa2_custom/ready_binding.cpp", "fa2_custom/ready_kernels.cu"],
    "b01_frontier_opt_ext": ["fa2_custom/frontier_opt_binding.cpp", "fa2_custom/frontier_opt_kernels.cu"],
    "offload_fa2_ext": ["fa2_custom/autotune_binding.cpp", "fa2_custom/autotune_kernels.cu"],
}
EXPORTS = {
    "producer_plan_ext": ["ProducerPlan"],
    "frontier_streamwrite_ext": ["write_frontier"],
    "fa2_streaming_consumer_ext": ["streaming_fwd", "publish_frontier", "launch_producer"],
    "b01_ready_ext": ["fwd", "info"],
    "b01_frontier_opt_ext": ["fwd", "info"],
    "offload_fa2_ext": ["fwd", "info"],
}
ABI_FIELDS = ("torch_version", "torch_cuda", "cxx11_abi", "python_version", "python_executable", "soabi", "ext_suffix")


def validate_abi(abi):
    if not isinstance(abi, dict) or any(key not in abi for key in ABI_FIELDS):
        raise B01Error("CUDA_ABI", "complete Torch/Python ABI descriptor required")
    if type(abi["cxx11_abi"]) is not bool or any(not isinstance(abi[k], str) or not abi[k] for k in ABI_FIELDS if k != "cxx11_abi"):
        raise B01Error("CUDA_ABI", "invalid ABI field types")
    if not re.fullmatch(r"\d+\.\d+", abi["torch_cuda"]):
        raise B01Error("CUDA_ABI", "CUDA-enabled Torch required")
    if not Path(abi["python_executable"]).is_absolute() or not abi["ext_suffix"].endswith(".so") or "/" in abi["ext_suffix"]:
        raise B01Error("CUDA_ABI", "Linux extension suffix and explicit model Python required")
    return dict(abi)


def assert_abi(expected, actual):
    validate_abi(expected)
    validate_abi(actual)
    if expected != actual:
        keys = [k for k in ABI_FIELDS if expected[k] != actual[k]]
        raise B01Error("CUDA_ABI", f"ABI mismatch: {', '.join(keys)}")


def current_abi(torch):
    """Explicit gate only. Reading these attributes does not initialize CUDA."""
    return validate_abi(dict(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                             cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                             python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                             # A venv may symlink its interpreter to the base
                             # Python. Keep the invoked environment path: using
                             # the resolved target silently drops its site-packages.
                             python_executable=str(Path(sys.executable).absolute()),
                             soabi=sysconfig.get_config_var("SOABI"),
                             ext_suffix=sysconfig.get_config_var("EXT_SUFFIX")))


def compatibility_generation():
    return canonical_hash({name: digest(ROOT / name) for name in
                           ("build.py", "setup_compat.py", "cache.py", "contract.py", "integrity.py", "_source_lock.py")})


def dry_run(arch, abi, clean_commit, cuda_home, cache_root, spec_path):
    """No subprocess, Torch import, CUDA discovery, download, or file mutation."""
    if not isinstance(arch, str) or arch not in ARCHES:
        raise B01Error("UnsupportedArchitecture", f"unsupported CUDA_DEVICE_ARCH {arch!r}")
    if not isinstance(clean_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", clean_commit):
        raise B01Error("LogicError", "full clean release commit SHA required")
    for name, path in (("CUDA_HOME", cuda_home), ("cache", cache_root), ("spec", spec_path)):
        if not Path(path).is_absolute():
            raise B01Error("PathError", f"{name} must be an explicit absolute external path")
        if name != "CUDA_HOME" and Path(path).resolve().is_relative_to(ROOT.parents[3]):
            raise B01Error("PathError", f"{name} must be outside the clean release checkout")
    spec = dict(schema=1, source_generation=source_generation(), compatibility_generation=compatibility_generation(),
                clean_commit=clean_commit, arch=arch, abi=validate_abi(abi), cuda_home=str(cuda_home),
                cxx_flags=CXX_FLAGS, nvcc_flags=NVCC_FLAGS, modules=MODULES)
    # This is a request identity; the cache key additionally binds the checked toolchain.
    request_id = canonical_hash(spec)
    return dict(validation="STATIC", toolchain="NOT_CHECKED", request_id=request_id, spec=spec,
                spec_path=str(spec_path), cache_root=str(cache_root),
                command=[abi["python_executable"], str(ROOT / "setup_compat.py"), "build_ext"],
                environment={"B01_BUILD_SPEC": str(spec_path), "B01_CACHE_ROOT": str(cache_root),
                             "CUDA_HOME": str(cuda_home), "TORCH_CUDA_ARCH_LIST": arch},
                build_gate="EXPLICIT_CUDA_CI_REQUIRED; no compilation was performed")


def check_toolchain(spec, runner=subprocess.run, compiler=None):
    """No device access. Must run explicitly on the target model environment."""
    for key in ("CC", "CUDAHOSTCXX", "CFLAGS", "CPPFLAGS", "CXXFLAGS", "LDFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TORCH_NVCC_FLAGS"):
        if os.environ.get(key):
            raise B01Error("CUDA_ABI", f"unrecorded compiler override {key}; clear it for this build")
    home = Path(spec["cuda_home"])
    nvcc = home / "bin/nvcc"
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        raise B01Error("MissingDependency", f"NVCC not executable: {nvcc}")
    candidates = [home / "include"] + sorted((home / "targets").glob("*/include"))
    required = ("cuda.h", "cuda_runtime.h", "cuda_bf16.h", "cuda_fp16.h", "cuda_fp8.h", "cuda/atomic")
    includes = [p for p in candidates if p.is_dir()]
    if any(not any((p / name).is_file() for p in includes) for name in required):
        raise B01Error("MissingDependency", "CUDA runtime/driver/BF16/FP8/atomic headers missing")
    libs = [home / "lib64", home / "lib"] + sorted((home / "targets").glob("*/lib"))
    # Stub is link-time only. Never add stubs to runtime LD_LIBRARY_PATH.
    link_dirs = [p for p in libs if p.is_dir()]
    link_dirs += [p / "stubs" for p in libs if (p / "stubs").is_dir()]
    if not any(list(p.glob("libcuda.so*")) for p in link_dirs):
        raise B01Error("MissingDependency", "CUDA driver link library/stub missing under CUDA_HOME")

    def run(args):
        try:
            result = runner(args, check=True, capture_output=True, text=True, timeout=30)
            return result.stdout + result.stderr
        except (OSError, subprocess.SubprocessError) as exc:
            raise B01Error("MissingDependency", f"toolchain command failed: {args[0]}: {exc}") from exc

    version = run([str(nvcc), "--version"])
    match = re.search(r"release (\d+)\.(\d+)", version)
    if not match:
        raise B01Error("CUDA_ABI", "unrecognized NVCC version")
    toolkit = tuple(map(int, match.groups()))
    torch_cuda = tuple(map(int, spec["abi"]["torch_cuda"].split(".")))
    if toolkit[0] != torch_cuda[0] or toolkit < torch_cuda:
        raise B01Error("CUDA_ABI", "toolkit major must match Torch CUDA; toolkit must not be older")
    gpu_codes = run([str(nvcc), "--list-gpu-code"])
    if ARCHES[spec["arch"]] not in re.findall(r"sm_\d+[a-z]?", gpu_codes):
        raise B01Error("UnsupportedArchitecture", f"NVCC does not list {ARCHES[spec['arch']]}")
    compiler = compiler or os.environ.get("CXX", "c++")
    compiler_path = shutil.which(compiler)
    if not compiler_path:
        raise B01Error("MissingDependency", f"C++ compiler missing: {compiler}")
    compiler_version = run([compiler, "--version"])
    # Lock external header bytes too: the selected Torch NVIDIA wheel headers
    # can differ across ABI environments even when the NVCC executable matches.
    headers = {str(path): digest(path) for include in includes
               for path in sorted(include.rglob("*")) if path.is_file()}
    return dict(nvcc=str(nvcc), nvcc_sha256=digest(nvcc), version=version, gpu_codes=gpu_codes,
                cxx=compiler, cxx_path=str(Path(compiler_path).resolve()), cxx_sha256=digest(compiler_path),
                cxx_version=compiler_version, includes=list(map(str, includes)),
                library_dirs=list(map(str, link_dirs)), header_sha256=headers)


def check_clean_checkout(commit, runner=subprocess.run):
    """Real-build gate only; a dry-run SHA is an unchecked caller declaration."""
    try:
        head = runner(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                      capture_output=True, text=True, timeout=30).stdout.strip()
        status = runner(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
                        check=True, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise B01Error("LogicError", "clean source checkout cannot be verified") from exc
    if head != commit or status:
        raise B01Error("LogicError", "build requires the exact declared clean release commit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=os.environ.get("CUDA_DEVICE_ARCH"), required="CUDA_DEVICE_ARCH" not in os.environ)
    parser.add_argument("--abi", type=Path, required=True)
    parser.add_argument("--clean-commit", required=True)
    parser.add_argument("--cuda-home", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--spec-path", required=True)
    args = parser.parse_args()
    try:
        plan = dry_run(args.arch, json.loads(args.abi.read_text()), args.clean_commit,
                       args.cuda_home, args.cache_root, args.spec_path)
        print(json.dumps(plan, indent=2))
    except (B01Error, OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
