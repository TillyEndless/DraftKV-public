"""Offline input, source, checkpoint and launch binding. No torch imports."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from .catalog import MODELS, ROUTE_MODES
from .contracts import BackboneError, external_path
from .sequence_parallel import attention_sp, local_heads, model_topology, topology_for


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify_files(root, locks):
    """Every imported source file must be within its declared external root."""
    root = Path(root).resolve()
    if not isinstance(locks, dict) or not locks:
        raise BackboneError("BLOCKED_SOURCE", "source_files SHA-256 inventory required")
    for relative, digest in locks.items():
        path = (root / relative).resolve()
        if (Path(relative).is_absolute() or not path.is_relative_to(root)
                or not path.is_file() or sha256(path) != digest):
            raise BackboneError("SOURCE_SHA_MISMATCH", str(path))


def source_inventory(root):
    """Read-only source audit helper; does not bless or rewrite an existing lock."""
    root = Path(root).resolve()
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*.py"))
            if not any(part in (".git", "__pycache__", ".venv") for part in path.relative_to(root).parts)}


def probe_environment(python, family="lingbot"):
    """Read installed distribution metadata in the model environment, without CUDA imports."""
    packages = ("torch", "flash-attn", "diffusers", "safetensors", "numpy", "Pillow",
                "easydict", "einops", "transformers", "tokenizers", "tqdm", "sentencepiece")
    if family == "solarwm":
        packages = ("torch", "flash-attn", "diffusers", "safetensors", "numpy", "Pillow", "einops", "transformers", "tokenizers", "peft", "imageio")
    if family == "longlive":
        packages += ("peft", "torchvision", "ftfy")
    source = ("import importlib.metadata as m,json,sys; "
              f"packages={packages!r}; "
              "available={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}; "
              "print(json.dumps({'python_tag':f'cpython-{sys.version_info.major}{sys.version_info.minor}',"
              "'python_executable':sys.executable,'python_prefix':sys.prefix,"
              "'packages':{name:available.get(name.lower().replace('_','-')) for name in packages}}))")
    result = subprocess.run([str(python), "-c", source], check=True, capture_output=True, text=True, timeout=30)
    report = json.loads(result.stdout)
    if family == "solarwm":
        lock = read_json(Path(__file__).resolve().parents[3] / "configs/backbones/solarwm_runtime_lock.json")
        check = """import hashlib, importlib.metadata as m, json, sys
lock=json.loads(sys.argv[1]); observed={}
for name, expected in lock['packages'].items():
    distribution=m.distribution(name)
    observed[name]={'version':distribution.version,'files':{
        relative:hashlib.sha256(distribution.locate_file(relative).read_bytes()).hexdigest()
        for relative in expected['files']}}
print(json.dumps(observed))
"""
        actual = subprocess.run([str(python), "-c", check, json.dumps(lock)],
                                check=True, capture_output=True, text=True, timeout=30)
        observed = json.loads(actual.stdout)
        if observed != lock["packages"]:
            raise BackboneError("SOURCE_ABI_DRIFT", "H3 native dependency source differs from the release lock")
        report["native_dependency_sources"] = observed
    missing = [name for name, version in report["packages"].items() if version is None]
    if missing:
        raise BackboneError("BLOCKED_DEPENDENCY", "missing model distributions: " + ", ".join(missing))
    if not str(report["packages"]["flash-attn"]).startswith("2."):
        raise BackboneError("BLOCKED_DEPENDENCY", "FlashAttention 2 is required")
    def executable(name):
        specified = os.environ.get(f"{name.upper()}_EXE")
        if specified:
            return os.path.isfile(specified) and os.access(specified, os.X_OK)
        return bool(shutil.which(name))
    if not executable("ffprobe") or not executable("ffmpeg"):
        raise BackboneError("BLOCKED_DEPENDENCY", "native video writer/validation requires ffmpeg and ffprobe (PATH or FFmpeg/FFprobe EXE overrides)")
    return report


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", f"{name} must be a positive integer")
    return value


def history_plan(spec, request, sp, config):
    """Size the original configurable pinned owner; never change its retrieval budget.

    Wan's VAE spatial stride is 8 and native attention patch is 2x2. The frozen
    output dimensions are checked again against real native tokens by the worker.
    Minimum24 preserves the LingBot helper reservation; LongLive requires12.
    Capacity otherwise follows the whole requested latent sequence.
    """
    width, height = request["output_width"], request["output_height"]
    if width % 16 or height % 16:
        raise BackboneError("BLOCKED_OUTPUT_SPEC", "native output must align to VAE/attention spatial stride16")
    minimum = 12 if spec.family == "longlive" else 24
    capacity = max(minimum, request["latent_frames"])
    tokens = spec.frame_tokens or (width // 16) * (height // 16)
    heads = local_heads(spec.heads, spec.family, sp)
    topology = topology_for(spec.family, sp, spec.heads)
    frame_bytes = spec.layers * 2 * tokens * heads * spec.head_dim * 2
    per_rank = capacity * frame_bytes
    rank_limit = config.get("pinned_bank_max_bytes_per_rank", per_rank)
    host_limit = config.get("pinned_bank_max_bytes_host", per_rank * sp)
    return {"capacity_frames": capacity, "frame_tokens": tokens, "layers": spec.layers,
            "heads_per_rank": heads, "sp_topology": topology,
            "paired_bf16_bytes_per_frame": frame_bytes,
            "required_bytes_per_rank": per_rank, "required_bytes_host": per_rank * sp,
            "limit_bytes_per_rank": rank_limit, "limit_bytes_host": host_limit,
            "owner": "original_configurable_pinned_frame_pool", "gpu_validated": False}


def _conditioning_binding(request, row, root):
    """Bind prepared conditioning to a frozen row or hashed preprocessing receipt.

    A receipt records source bytes and transformation provenance. This verifies
    provenance/bytes, not the correctness of an unreviewed preprocessing program.
    Formal benchmark acceptance remains owned by the Dataset/Evaluator protocol.
    """
    binding = row.get("native_input")
    if binding is None and request.get("conditioning_receipt"):
        path = external_path(request["conditioning_receipt"], root)
        if not path.is_file() or sha256(path) != request.get("conditioning_receipt_sha256"):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "conditioning receipt missing or SHA mismatch")
        receipt = read_json(path)
        for field in ("case_id", "dataset_manifest_sha256", "dataset_row_sha256"):
            if receipt.get(field) != request.get(field):
                raise BackboneError("BLOCKED_INPUT_PROTOCOL", "conditioning receipt has a different frozen case binding")
        source_assets = receipt.get("source_assets")
        if not isinstance(source_assets, list) or not source_assets or not receipt.get("transformation"):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "conditioning receipt requires source hashes and transformation provenance")
        for asset in source_assets:
            source = external_path(asset["path"], root)
            if not source.is_file() or sha256(source) != asset.get("sha256"):
                raise BackboneError("BLOCKED_INPUT_PROTOCOL", "conditioning source asset missing or SHA mismatch")
        binding = receipt.get("native_input")
    if not isinstance(binding, dict):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "prepared conditioning is not bound to a frozen row/receipt")
    for field in ("asset_sha256", "camera_coordinate_system", "prompts", "frames", "native_frames",
                  "width", "height", "output_width", "output_height", "fps", "seed", "image_conditioning",
                  "input_conversion", "output_spec_provenance", "camera_policy"):
        if binding.get(field) != request.get(field):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", f"prepared {field} differs from frozen conditioning binding")
    if request.get("input_mode") == "bench1_t2v":
        for field in ("latent_frames", "vae_decode_method", "prompt_switch_frames", "prompt_execution_schedule", "prompt_timeline"):
            if binding.get(field) != request.get(field):
                raise BackboneError("BLOCKED_OUTPUT_SPEC", "native T2V " + field + " differs from frozen output binding")
    if binding.get("padding_policy") != request.get("padding_policy"):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "padding policy differs from frozen conditioning binding")


def resolve_request(config, root, group):
    """Read a normalized, explicitly frozen per-case model input contract.

    This is an input binding, not a replacement Dataset protocol. The frozen
    row owns prompt count and frame count. No six-prompt or 480-frame constant.
    """
    path = external_path(config["request"], root)
    if not path.is_file() or sha256(path) != config.get("request_sha256"):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "request file missing or SHA mismatch")
    request = read_json(path)
    if request.get("execution_scope") not in ("diagnostic", "release_candidate"):
        raise BackboneError("BLOCKED_DATASET_BINDING", "adapter requests require explicit diagnostic/release_candidate scope")
    if request.get("execution_scope") == "release_candidate" and request.get("protocol") != group.protocol.to_dict():
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "release candidate requires the exact frozen Dataset protocol")
    expected = ROUTE_MODES[(group.benchmark, group.backbone)].value
    if request.get("input_mode") != expected:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", f"this route requires {expected}")
    if group.benchmark == "bench2" and request.get("execution_scope") == "release_candidate":
        from draftkv.benchmarks.datasets import DatasetError
        from draftkv.benchmarks.orbia_requests import validate_request_binding
        if request.get("backbone") != group.backbone:
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "ORBIA prepared request belongs to a different native model")
        try:
            validate_request_binding(request, root)
        except (DatasetError, KeyError, OSError, ValueError) as exc:
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", str(exc)) from exc
    freeze = external_path(request["dataset_manifest"], root)
    if not freeze.is_file() or sha256(freeze) != request.get("dataset_manifest_sha256"):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "frozen dataset manifest missing or SHA mismatch")
    manifest = read_json(freeze)
    rows = manifest.get("cases") if isinstance(manifest, dict) else manifest
    if not isinstance(rows, list):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "frozen manifest must expose cases for an exact case binding")
    matches = [row for row in rows if row.get("case_id", row.get("sample_id")) == request.get("case_id")]
    if len(matches) != 1:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "case not uniquely present in frozen manifest")
    row = matches[0]
    for field in ("selected_case_ids", "frozen_case_ids"):
        selected = manifest.get(field) if isinstance(manifest, dict) else None
        if selected is not None and (not isinstance(selected, list) or request["case_id"] not in selected):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "case excluded by frozen dataset selection")
    row_digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if row_digest != request.get("dataset_row_sha256"):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "frozen row binding changed")
    prompts = request.get("prompts")
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "explicit nonempty prompts required")
    original_prompts = row.get("prompts", [row["prompt"]] if "prompt" in row else None)
    if original_prompts is not None and prompts != original_prompts:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "prompt schedule differs from frozen row")
    frames = _positive(request.get("frames"), "frames")
    if row.get("frame_count", frames) != frames:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "frame count differs from frozen row")
    for field in ("width", "height", "fps", "output_width", "output_height"):
        _positive(request.get(field), field)
    if type(request.get("seed")) is not int:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "explicit seed required")
    if request["fps"] != 16:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "these native configurations emit 16 FPS")
    request["dataset_manifest"] = str(freeze)
    if request.get("input_conversion") is not None:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "generated-frame/static-camera T2V conversion is retired; use the actual frozen modality")
    if group.backbone == "SolarWM33B":
        if request.get("execution_scope") != "release_candidate" or len(prompts) != 1 or request["seed"] < 0:
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "H3 requires the frozen official ORBIA request")
        _conditioning_binding(request, row, root)
        request.update(image=str(external_path(request["image"], root)),
                       camera_dir=str(external_path(request["camera_dir"], root)),
                       dataset_root=str(external_path(request["dataset_root"], root)))
        return request
    if group.backbone == "LongLive1.3B":
        if (len(prompts) != 6 or any(request.get(key) is not None for key in
                ("image", "initial_image", "adapter_image", "clean_latent_prefix", "camera", "camera_dir"))
                or request.get("image_conditioning") not in (None, "none") or request.get("asset_sha256", {})
                or request["width"] % 16 or request["height"] % 16
                or (request["output_width"], request["output_height"]) != (request["width"], request["height"])):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "LongLive Bench1 is native text-only T2V; image/prefix/camera input is forbidden")
        from draftkv.benchmarks.worldattention import verify_manifest, validate_schedule
        from draftkv.benchmarks.datasets import DatasetError
        try:
            verify_manifest(freeze)
            validate_schedule(request)
        except DatasetError as error:
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", str(error)) from error
        if request.get("prompt_timeline") != row.get("timeline"):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "nominal timeline differs from frozen row")
        if request.get("sample_shift", 5.0) != 5.0 or request.get("offload_model", True) is not True:
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "LongLive native binding uses shift5 and explicit model/conditioning offload")
        native_frames = _positive(request.get("native_frames"), "native_frames")
        latent_frames = request.get("latent_frames")
        # The native decoder owns temporal chunk boundaries. Its encoded frame
        # count is independently frozen, never inferred using a full-decode
        # formula or borrowed from the TI manifest.
        if type(latent_frames) is not int or latent_frames <= 0 or latent_frames % 3:
            raise BackboneError("BLOCKED_OUTPUT_SPEC", "native T2V requires separately frozen positive C3 latent_frames; do not borrow the TI961 specification")
        if native_frames != frames:
            raise BackboneError("BLOCKED_OUTPUT_SPEC", "native T2V frames/native_frames must explicitly bind the untrimmed native decoder output")
        if request.get("vae_decode_method", "decode_to_pixel_chunk") not in ("decode_to_pixel", "decode_to_pixel_chunk"):
            raise BackboneError("BLOCKED_OUTPUT_SPEC", "native T2V decoder must be an explicitly source-bound author method")
        if request.get("output_spec_provenance") != "frozen_native_t2v":
            raise BackboneError("BLOCKED_OUTPUT_SPEC", "source prompts do not freeze native frames; a separate frozen_native_t2v output binding is required")
        _conditioning_binding(request, row, root)
        request.update(latent_frames=latent_frames, sample_shift=5.0, offload_model=True,
                       image_conditioning="none")
        return request
    if len(prompts) != 1:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "native LingBot accepts one prompt; scheduled prompting needs an audited binding")
    image = external_path(request["image"], root)
    camera = external_path(request["camera_dir"], root)
    assets = {"image": image, "poses": camera / "poses.npy", "intrinsics": camera / "intrinsics.npy"}
    hashes = request.get("asset_sha256", {})
    for name, asset in assets.items():
        if not asset.is_file() or sha256(asset) != hashes.get(name):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", f"{name} missing or SHA mismatch")
    provenance_allowed = ("frozen_manifest",)
    if request.get("camera_coordinate_system") != "opencv_c2w" or request.get("camera_provenance") not in provenance_allowed:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "authentic frozen OpenCV C2W conditioning required")
    # A caller must supply the author's padded camera explicitly and retain
    # provenance. No synthetic camera or silent frame extension is performed.
    native_frames = _positive(request.get("native_frames"), "native_frames")
    latent_frames = (native_frames - 1) // 4 + 1
    if (native_frames - 1) % 4 or latent_frames % 4 or native_frames < frames:
        raise BackboneError("BLOCKED_OUTPUT_SPEC", "LingBot must produce complete native C4 latent chunks")
    if native_frames != frames and request.get("padding_policy") != "hold_last_camera_trim_video":
        raise BackboneError("BLOCKED_OUTPUT_SPEC", "explicit frozen tail padding/trim binding required")
    _conditioning_binding(request, row, root)
    request.update(image=str(image), camera_dir=str(camera), latent_frames=latent_frames,
                   sample_shift=request.get("sample_shift", 5.0 if group.backbone == "LingBot1.3B" else 10.0),
                   offload_model=request.get("offload_model", True))
    if (type(request["sample_shift"]) not in (int, float) or not math.isfinite(request["sample_shift"])
            or request["sample_shift"] <= 0 or type(request["offload_model"]) is not bool):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "invalid native parameter binding")
    return request


DRAFTKV_SP8_BF16_PROFILE = "lingbot14b-sp8-bf16-stream-20261102"


def validate_loading_profile(config, group, sp):
    """Validate optional profiling; the backbone owns ordinary initialization."""
    selected = config.get("draftkv_loading_profile")
    if selected is None:
        return None
    if selected != DRAFTKV_SP8_BF16_PROFILE:
        raise BackboneError("BLOCKED_CONFIGURATION", "unknown DraftKV loading profile")
    if (group.benchmark, group.backbone, group.algorithm, sp) != ("bench2", "LingBot14B", "draftkv", 8):
        raise BackboneError("BLOCKED_CONFIGURATION", "BF16 stream profile is only for Bench2 LingBot14B DraftKV SP8")
    return selected


def make_plan(root, group, sp, config, checkpoint, run_dir):
    from draftkv.algorithms.draftkv.experiment import experiment_scope
    with experiment_scope(config.get('algorithm_plugin', {}).get('config', {}), group.backbone):
        return _make_plan(root, group, sp, config, checkpoint, run_dir)


def _make_plan(root, group, sp, config, checkpoint, run_dir):
    loading_profile = validate_loading_profile(config, group, sp)
    spec = MODELS[group.backbone]
    if spec.sp_status(sp, group.algorithm) == "UNSUPPORTED":
        raise BackboneError("UNSUPPORTED", f"{group.backbone}/{group.algorithm}/SP{sp}")
    builtin = group.algorithm in ("native", "draftkv")
    if group.algorithm == "native" and config.get("algorithm_plugin"):
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "Native may not be replaced by an algorithm plugin")
    if not builtin and not config.get("algorithm_plugin"):
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", f"no audited {group.algorithm} plugin bound")
    request = resolve_request(config, root, group)
    from draftkv.release.audit import (release_source_files, require_release_sources,
                                      validate_plugin_binding)
    official_release = group.release_target and request.get("execution_scope") == "release_candidate"
    adapter_files = release_source_files(root, require_complete=official_release)
    from draftkv.runtime.state import git_sha
    try:
        repo_commit = git_sha(root)
    except ValueError:
        repo_commit = None  # Explicitly unknown for non-Git historical fixtures/snapshots.
    source = external_path(config["source_root"], root)
    lock_path = external_path(config["source_lock"], root)
    lock = read_json(lock_path)
    lock = lock.get(group.backbone, lock)
    verify_files(source, lock["files"])
    native_root = (source / lock.get("native_subdir", ".")).resolve()
    if not native_root.is_relative_to(source):
        raise BackboneError("BLOCKED_SOURCE", "native root escapes source tree")
    stack = (("wan/modules/causal_model.py", "wan/modules/attention.py", "utils/wan_wrapper.py",
              "utils/scheduler.py", "pipeline/causal_inference.py") if spec.family == "longlive" else
             ("generate.py", "wan/image2video.py", "wan/modules/model_fast.py",
              "wan/modules/attention.py", "wan/distributed/sequence_parallel.py"))
    if spec.family == "solarwm":
        stack = ("model.py", "sgf_attention.py", "sgf_rollout.py", "full_inference.py",
                 "full_length.py", "torch_prope.py", "torch_geometry.py", "distributed.py", "optional.py", "weights.py")
    for relative in stack:
        required = str((native_root / relative).relative_to(source))
        if required not in lock["files"]:
            raise BackboneError("BLOCKED_SOURCE", f"unlocked native call stack file: {required}")
    import_roots = [str((source / path).resolve()) for path in lock["import_subdirs"]]
    if any(not Path(path).is_relative_to(source) for path in import_roots):
        raise BackboneError("BLOCKED_SOURCE", "import roots escape the source tree")
    provider = None
    if group.algorithm == "draftkv" and not config.get("algorithm_plugin"):
        if spec.family == "longlive":
            from .longlive_provider import resolve_provider
            provider = resolve_provider(config.get("longlive_provider"), root)
        elif lock.get("binding") != "hierarchical_v1_original_b01":
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "a legacy DraftMap/Native source cannot substitute for V1+B01")
    history = (history_plan(spec, request, sp, config)
               if group.algorithm == "draftkv" else None)
    checkpoint = external_path(checkpoint, root)
    if not checkpoint.exists():
        raise BackboneError("BLOCKED_CHECKPOINT", str(checkpoint))
    assets_dir = external_path(config["assets_dir"], root)
    shared_assets = (("transformer", "text_encoder", "tokenizer", "processor", "vae") if spec.family == "solarwm" else
                     ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl"))
    for asset in shared_assets:
        if not (assets_dir / asset).exists() and not (checkpoint / asset).exists():
            raise BackboneError("BLOCKED_DEPENDENCY", f"missing native shared asset {asset}")
    lora = None
    if spec.family == "longlive":
        if not checkpoint.is_file():
            raise BackboneError("BLOCKED_CHECKPOINT", "LongLive base generator .pt required")
        lora = external_path(config["lora_checkpoint"], root) if config.get("lora_checkpoint") else None
        if not lora or not lora.is_file():
            raise BackboneError("BLOCKED_CHECKPOINT", "LongLive native release requires the original base plus LoRA checkpoint")
    elif spec.family == "solarwm":
        from .solarwm_checkpoint import release_binding, validate_checkpoint
        _, expected_step = release_binding(root)
        validate_checkpoint(checkpoint, expected_step)
    else:
        dit = checkpoint / "transformers" if (checkpoint / "transformers").is_dir() else checkpoint
        if not dit.is_dir() or not list(dit.glob("*.safetensors")):
            raise BackboneError("BLOCKED_CHECKPOINT", "native safetensors checkpoint not found")
    python = external_path(config["python"], root, resolve_symlinks=False)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise BackboneError("BLOCKED_DEPENDENCY", "external model interpreter missing or not executable")
    environment = probe_environment(python, spec.family)
    plugin = config.get("algorithm_plugin")
    if plugin:
        plugin = dict(plugin)
        plugin["root"] = str(external_path(plugin["root"], root))
        verify_files(plugin["root"], plugin["files"])
        if plugin.get("entry") not in plugin["files"]:
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "plugin entry must be source locked")
    source_audit = None
    if official_release:
        try:
            source_audit = require_release_sources(root, (group,))
            validate_plugin_binding(root, group, plugin, expected_files=adapter_files)
        except ValueError as exc:
            raise BackboneError("BLOCKED_RELEASE_BINDING", str(exc)) from exc
    kernel_dir = None
    kernel_locks = {}
    canonical_b01 = group.algorithm == "draftkv" and plugin and plugin.get("entry") == "src/draftkv/algorithms/draftkv/entry.py"
    if canonical_b01:
        from draftkv.streaming.binding import NativeBinding
        settings = dict(plugin.get("config", {}))
        from draftkv.algorithms.draftkv.position import require_policy, require_packing_policy
        from draftkv.algorithms.draftkv.entry import configure_retrieval
        settings = configure_retrieval(group.backbone, settings)
        settings["position_policy"] = require_policy(settings)
        settings["packing_policy"] = require_packing_policy(settings)
        plugin["config"] = settings
        binding = NativeBinding.read(settings, root)
        # Use the same strict native verification as the worker: Python-only
        # bridge upgrades may reuse byte-verified compiled producer/consumer
        # modules. The legacy cache check rejects that supported transition.
        binding.verify_build_cache()
    if group.algorithm == "draftkv" and not canonical_b01:
        if request.get("execution_scope") == "release_candidate":
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS",
                                "DraftKV releases require the canonical Chrono-Compact plugin; regenerate the source-bound config")
        kernel_dir = external_path(config["kernel_dir"], root)
        kernel_locks = config.get("kernel_sha256", {})
        if not any(name.startswith("producer_plan_ext") for name in kernel_locks) or not any(name.startswith("offload_fa2_ext") for name in kernel_locks):
            raise BackboneError("BLOCKED_DEPENDENCY", "original B01 producer and consumer binary hashes required")
        verify_files(kernel_dir, kernel_locks)
        if any(environment["python_tag"] not in name for name in kernel_locks):
            raise BackboneError("BLOCKED_DEPENDENCY", "B01 extension Python ABI differs from the model interpreter")
        abi = config.get("kernel_abi", {})
        if (abi.get("python_tag") != environment["python_tag"]
                or abi.get("torch") != environment["packages"]["torch"]
                or abi.get("cuda") != str(environment["packages"]["torch"]).split("+")[-1]
                or not isinstance(abi.get("sm"), list) or not abi["sm"]):
            raise BackboneError("BLOCKED_DEPENDENCY", "B01 Python/Torch/CUDA build metadata missing or incompatible")
    from draftkv.runtime.config import asset_sha256, json_sha256
    checkpoint_hashes = {"base": asset_sha256(checkpoint)}
    extra = {}
    if spec.family == "solarwm":
        if group.algorithm != "native" and not plugin:
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "H3 requires shared canonical plugins")
        silence = external_path(config["silence_path"], root)
        if not silence.is_file():
            raise BackboneError("BLOCKED_DEPENDENCY", "H3 silence asset missing")
        checkpoint_hashes.update(native_base=asset_sha256(assets_dir), silence=sha256(silence))
        extra = {"native_model_config": read_json(root / "configs/backbones/solarwm_model.json"),
                 "silence_path": str(silence)}
    if lora:
        checkpoint_hashes["lora"] = asset_sha256(lora)
    from .budgets import author_budget_policy, effective_budget
    preset = (plugin or {}).get("config", {}).get("preset") if group.algorithm in ("deep-forcing", "future-forcing", "worldtrace-landmark") else None
    budget = effective_budget(spec, group.algorithm, preset=preset).to_dict()
    policy = author_budget_policy(spec, group.algorithm, preset=preset)
    if config.get("effective_budget", budget) != budget:
        raise BackboneError("BLOCKED_CONFIGURATION", "effective budget differs from release binding")
    if config.get("author_budget_policy", policy) != policy:
        raise BackboneError("BLOCKED_CONFIGURATION", "author budget policy differs from release binding")
    plan = {"schema_version": 1, **group.to_dict(), **extra, "sp": sp, "repo_commit": repo_commit,
            "effective_budget": budget, "author_budget_policy": policy,
            "sp_mode": topology_for(spec.family, sp, spec.heads),
            "attention_sp": attention_sp(spec.family, sp, spec.heads),
            "distributed_topology": model_topology(spec.family, spec.heads, sp), "repo_root": str(root),
            "checkpoint_hashes": checkpoint_hashes, "checkpoint_sha256": json_sha256(checkpoint_hashes),
            "request": request, "source_root": str(source), "native_root": str(native_root),
            "source_files": lock["files"], "import_roots": import_roots,
            "adapter_source_files": adapter_files, "source_binding_audit": source_audit,
            "checkpoint": str(checkpoint), "assets_dir": str(assets_dir), "python": str(python),
            "lora_checkpoint": str(lora) if lora else None,
            "environment_metadata": environment,
            "history_plan": history, "longlive_provider": provider,
            "kernel_dir": str(kernel_dir) if kernel_dir else None, "kernel_sha256": kernel_locks,
            "kernel_abi": config.get("kernel_abi") if kernel_dir else None,
            "algorithm_plugin": plugin, "run_dir": str(run_dir),
            "output": str(Path(run_dir) / "native.mp4"), "release_e2e": False}
    if loading_profile is not None:
        plan["draftkv_loading_profile"] = loading_profile
    if canonical_b01:
        plan.update(binding.host_pages_metadata)
    if group.algorithm in ("deep-forcing", "future-forcing"):
        from draftkv.algorithms.forcing.geometry import native_geometry, output_request
        plan["native_output_geometry"] = native_geometry(plan)
        plan["history_plan"] = history_plan(spec, output_request(plan), sp, config)
    if canonical_b01:
        plan['attention_position_policy'] = settings['position_policy']
        plan['retrieval_position_policy'] = 'native_post_rope'
        dimensions = 32 if spec.family == 'solarwm' else 44
        elements = request['latent_frames'] * spec.layers * history['frame_tokens'] * history['heads_per_rank'] * dimensions
        plan['history_plan']['additional_raw_temporal_key_storage'] = {
            'device': 'cpu', 'real_dimensions': dimensions,
            'bytes_bf16': elements * 2, 'bytes_fp32': elements * 4,
            'dtype': 'preserve_native_raw_key_dtype', 'pinned': False}
    return plan
