"""Strict profile binding shared by deployment preflight and real attention.

Reading historical recipes is never device acceptance. The GPU call site must
provide a freshly measured complete request, rather than echoing saved JSON.
"""

from pathlib import Path

from .profiles import (ProfileError, check_config, checked, digest, load_catalog, read_json,
                       request_family, require, select_profile)


def _path(value, root):
    require(isinstance(value, str) and bool(value), "BLOCKED_PROFILE", "explicit profile artifact path required")
    path = Path(value).expanduser()
    return (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()


def verify_scheduler_lease(path, request, purpose="profile_autotune"):
    """A profile CLI can consume an allocation, but cannot allocate a device."""
    from datetime import datetime, timezone
    require(isinstance(path, str) and bool(path), "GPU_GATE_REQUIRED", "unique Scheduler lease required")
    lease = read_json(path)
    try:
        expires = datetime.fromisoformat(lease["expires_utc"].replace("Z", "+00:00"))
        valid = (lease["issuer"] == "unique-gpu-scheduler" and lease["status"] == "allocated"
                 and bool(lease["job_id"]) and purpose in lease["purposes"]
                 and len(lease["device_uuids"]) == len(set(lease["device_uuids"]))
                 and set(lease["device_uuids"]) == {rank["device"]["uuid"] for rank in request["ranks"]}
                 and expires.tzinfo is not None and expires > datetime.now(timezone.utc))
    except (KeyError, ValueError, TypeError):
        valid = False
    require(valid, "GPU_GATE_REQUIRED", "Scheduler lease expired or job/purpose/physical UUID differs")
    return lease


def verify_deployment(binding, root, *, selected=None, gpu=False, actual_request=None):
    """Verify one externally bound profile without probing or initializing CUDA.

    ``selected`` is the deployment inventory in physical rank order. ``gpu``
    additionally requires the measured full device/ABI/tensor request and two
    independently poisoned Golden replays. There is no SM-only or Native route.
    """
    require(isinstance(binding, dict) and bool(binding), "GPU_GATE_REQUIRED",
            "hardware_profile_binding required: catalog, request, artifact_root")
    require(set(binding) <= {"catalog", "request", "artifact_root", "asset_roots"}
            and {"catalog", "request", "artifact_root"} <= set(binding),
            "BLOCKED_PROFILE", "unknown or missing hardware profile binding fields")
    catalog_path = _path(binding["catalog"], root)
    request_path = _path(binding["request"], root)
    artifact_root = _path(binding["artifact_root"], root)
    catalog = load_catalog(catalog_path)
    request = checked(read_json(request_path), "request")
    name = request_family(request)
    assets = None
    if binding.get("asset_roots"):
        assets = read_json(_path(binding["asset_roots"], root))
        require(isinstance(assets, dict) and all(isinstance(k, str) for k in assets),
                "BLOCKED_PROFILE", "asset_roots must be an external path mapping")
        assets = {k: str(_path(v, root)) for k, v in assets.items()}
    if selected is not None:
        require(len(selected) == len(request["ranks"]), "UNSUPPORTED_GPU", "selected rank count differs from profile")
        for observed, rank in zip(selected, request["ranks"]):
            device = rank["device"]
            require(observed.get("uuid") == device["uuid"] and observed.get("name") == device["name"],
                    "UNSUPPORTED_GPU", "actual physical UUID/name differs from requested profile")
            capability = observed.get("compute_capability")
            if capability is not None:
                require(str(capability).replace(".", "") == str(device["sm"]),
                        "UNSUPPORTED_GPU", "actual device SM differs from profile")
            if observed.get("driver_version") is not None:
                require(observed["driver_version"] == request["abi"]["driver"],
                        "BLOCKED_ABI", "actual device driver differs from profile")
    if gpu or actual_request is not None:
        require(actual_request is not None, "GPU_GATE_REQUIRED",
                "fresh device/ABI/QKV capture required at real attention binding")
        checked(actual_request, "request")
        require(actual_request == request, "BLOCKED_ABI",
                "live physical identity, ABI, source/kernel, QKV strides, geometry or SP differs")
    try:
        result = select_profile(catalog, request, artifact_root, gpu=gpu, assets=assets)
    except ProfileError as exc:
        # The checked catalog contains no accepted binding on a fresh host.
        # Report an unperformed hardware gate, never manufacture a profile.
        if exc.code == "BLOCKED_PROFILE" and not any(p["family"] == name for p in catalog["profiles"]):
            raise ProfileError("GPU_GATE_REQUIRED", f"{name}: {exc}") from exc
        raise
    require(result["profile"]["config"]["producer"] == "cpp_dma", "GPU_GATE_REQUIRED",
            "historical H200 Python/framewise DMA recipe is not the P5 original C++ ProducerPlan chain")
    return {**result, "request_sha256": digest(request), "catalog_sha256": catalog["sha256"]}


def verify_plan_profile(profile, spec, *, require_geometry=True):
    """Assert the actual native plan implements every accepted kernel parameter."""
    profile = checked(profile, "profile")
    config = profile["config"]
    check_config(config, profile["key"])
    require(config["producer"] == "cpp_dma" and config["consumer_start"] == "initialized",
            "GPU_GATE_REQUIRED", "P5 requires original C++ DMA and initialized frontier")
    supported = {"ready": "ready", "frontier_opt_v9": "frontier_opt", "autotune": "autotune"}
    require(config["entrypoint"] in supported, "UNSUPPORTED_SHAPE",
            "accepted profile requires an original consumer absent from the current PlanSpec ABI")
    layout = {"grouped": "head_group_major"}.get(
        getattr(spec, "layout", "head_group_major"), getattr(spec, "layout", "head_group_major"))
    require(spec.consumer == supported[config["entrypoint"]]
            and spec.head_group == config["head_group"] and spec.gate_tiles == config["gate_tiles"]
            and spec.block_n == config["block_n"] and getattr(spec, "splits", 1) == config["splits"]
            and getattr(spec, "block_m", 64) == config["block_m"] and layout == config["layout"],
            "BLOCKED_PROFILE", "actual ProducerPlan/Streaming FA2 parameters differ from accepted profile")
    for key in ("dispatch", "host_pages"):
        # Legacy specs implement all/pytorch; a missing field cannot be used to
        # silently execute an early or huge-page scheduling profile.
        require(getattr(spec, key, {"dispatch": "all", "host_pages": "pytorch"}[key]) == config[key],
                "BLOCKED_PROFILE", "actual native plan scheduling/page policy differs from accepted profile")
    if spec.consumer == "frontier_opt":
        require(spec.variant == 9, "BLOCKED_PROFILE", "frontier_opt_v9 requires native variant 9")
    require(str(spec.arch).replace(".", "") == str(profile["key"]["ranks"][0]["device"]["sm"]),
            "UNSUPPORTED_GPU", "PlanSpec architecture differs from exact profile")
    # Emit *observed PlanSpec geometry*, not fabricated GPU tensor metadata, so
    # a failed warmup/partial-chunk request is actionable while still blocked.
    actual_shape = {
        "heads": int(spec.heads), "kv": int(spec.kv_tokens),
        "frame_tokens": int(spec.frame_tokens),
        "history_tokens": int(len(spec.frame_ids) * spec.frame_tokens)}
    geometry_keys = tuple(actual_shape)
    expected_shapes = [{
        "rank": index,
        **{key: rank["shape"][key] for key in geometry_keys}}
        for index, rank in enumerate(profile["key"]["ranks"])]
    if require_geometry and any(any(row[key] != actual_shape[key] for key in geometry_keys)
           for row in expected_shapes):
        from json import dumps
        raise ProfileError("UNSUPPORTED_SHAPE",
            "native plan geometry differs from exact accepted per-rank profile; "
            + "ACTUAL_PLAN_SPEC_ONLY=" + dumps(actual_shape, sort_keys=True)
            + "; ACCEPTED_PROFILE_RANK_SHAPES=" + dumps(expected_shapes, sort_keys=True)
            + "; full live CUDA QKV/strides/physical-device request not inferred")
    return profile


def capture_actual_request(*, q, k, v, frame_tokens, history_tokens,
                           source_sha256, kernel_sha256, parallel, torch_module=None):
    """Capture a live BSHD model seam on an authorized GPU worker only.

    Source/kernel hashes come from the already verified canonical native cache;
    tensor geometry, driver, toolkit, device and Python/Torch ABI are read fresh.
    All SP ranks must enter this function together. No saved request is read.
    """
    import csv
    import io
    import os
    import platform
    import re
    import subprocess
    import sysconfig

    if torch_module is None:
        import torch as torch_module
    t = torch_module
    require(q.is_cuda and k.is_cuda and v.is_cuda and q.device == k.device == v.device,
            "GPU_GATE_REQUIRED", "live same-device native Q/K/V required")
    require(q.ndim == k.ndim == v.ndim == 4 and tuple(k.shape) == tuple(v.shape)
            and q.dtype == k.dtype == v.dtype == t.bfloat16,
            "UNSUPPORTED_SHAPE", "capture expects native BSHD BF16 Q/K/V")
    require(q.shape[0] == k.shape[0] and q.shape[2:] == k.shape[2:],
            "UNSUPPORTED_SHAPE", "native Q/K/V batch/head geometry differs")
    props = t.cuda.get_device_properties(q.device)
    uuid = getattr(props, "uuid", "")
    if isinstance(uuid, bytes):
        uuid = uuid.decode("ascii")
    uuid = str(uuid)
    # PyTorch versions expose either the bare CUDA UUID or nvidia-smi's GPU-
    # spelling. Accept only the exact UUID syntax, then join physical inventory.
    if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
        uuid = "GPU-" + uuid
    require(bool(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid)),
            "GPU_GATE_REQUIRED", "CUDA properties must expose physical UUID")

    def run(argv):
        try:
            return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=15).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProfileError("GPU_GATE_REQUIRED", f"live identity probe failed: {argv[0]}") from exc

    inventory = run(["nvidia-smi", "--query-gpu=uuid,name,pci.bus_id,pci.device_id,driver_version",
                     "--format=csv,noheader,nounits"])
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(inventory))]
    matches = [row for row in rows if len(row) == 5 and row[0] == uuid]
    require(len(matches) == 1, "GPU_GATE_REQUIRED", "no unique physical UUID/PCI identity join")
    _, name, pci_bus, pci_device, driver = matches[0]
    nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc") if os.environ.get("CUDA_HOME") else "nvcc"
    version = re.search(r"release (\d+\.\d+)", run([nvcc, "--version"]))
    require(version is not None, "BLOCKED_ABI", "unrecognized installed Toolkit version")
    shape = {"batch": int(q.shape[0]), "q": int(q.shape[1]), "kv": int(k.shape[1]),
             "heads": int(q.shape[2]), "kv_heads": int(k.shape[2]), "dim": int(q.shape[3]),
             "dtype": "bf16", "causal": False, "layout": "BSHD",
             "q_strides": list(q.stride()), "k_strides": list(k.stride()), "v_strides": list(v.stride()),
             "history_tokens": history_tokens, "frame_tokens": frame_tokens}
    rank = {"device": {"name": name, "uuid": uuid, "pci_bus_id": pci_bus,
                       "pci_device_id": pci_device, "sm": props.major * 10 + props.minor,
                       "sm_count": props.multi_processor_count, "memory_bytes": props.total_memory},
            "shape": shape}
    abi = {"driver": driver, "toolkit": version.group(1), "torch": str(t.__version__),
           "torch_cuda": str(t.version.cuda), "python_abi": sysconfig.get_config_var("SOABI"),
           "cxx11_abi": bool(t._C._GLIBCXX_USE_CXX11_ABI), "platform": platform.platform()}
    require(isinstance(parallel, dict) and type(parallel.get("sp")) is int,
            "UNSUPPORTED_SHAPE", "explicit actual SP contract required")
    records = [{"rank": rank, "abi": abi}]
    if parallel["sp"] > 1:
        dist = t.distributed
        require(dist.is_initialized() and dist.get_world_size() == parallel["sp"],
                "GPU_GATE_REQUIRED", "live SP process group differs from profile capture")
        records = [None] * parallel["sp"]
        dist.all_gather_object(records, {"rank": rank, "abi": abi})
    require(all(record["abi"] == abi for record in records), "BLOCKED_ABI", "SP rank ABIs differ")
    request = {"ranks": [record["rank"] for record in records], "abi": abi,
               "b01": {"source_sha256": source_sha256, "kernel_sha256": kernel_sha256},
               "parallel": dict(parallel)}
    request_family(request)
    return request
