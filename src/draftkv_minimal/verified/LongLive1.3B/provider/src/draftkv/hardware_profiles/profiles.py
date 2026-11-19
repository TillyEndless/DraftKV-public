"""Exact physical-device/shape profiles. Importing this module never probes CUDA."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .schema import validate


class ProfileError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise ProfileError(code, message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def seal(value):
    value = deepcopy(value)
    value.pop("sha256", None)
    return {**value, "sha256": digest(value)}


def checked(value, kind):
    try:
        validate(value, kind)
        if kind in {"catalog", "profile"}:
            require(seal(value)["sha256"] == value["sha256"], "BLOCKED_PROFILE", "profile SHA256 mismatch")
        return deepcopy(value)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError("BLOCKED_PROFILE", str(exc)) from exc


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + x)))
    except (OSError, ValueError) as exc:
        raise ProfileError("BLOCKED_PROFILE", f"cannot read JSON: {path}: {exc}") from exc


def portable(path):
    require(isinstance(path, str) and bool(path) and "\\" not in path and ":" not in path,
            "BLOCKED_PROFILE", "artifact path must be repo-relative; configure its root externally")
    relative = PurePosixPath(path)
    require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == path,
            "BLOCKED_PROFILE", "nonportable artifact path")
    return relative


def verify_artifact(record, root, code="BLOCKED_PROFILE"):
    portable(record["path"])
    base = Path(root).resolve()
    path = (base / record["path"]).resolve()
    require(path.is_relative_to(base), code, "artifact escapes configured root")
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProfileError(code, f"unreadable required artifact: {record['path']}") from exc
    require(actual == record["sha256"], code, f"artifact SHA256 mismatch: {record['path']}")
    return path


def source_digest(records):
    require(len({r["path"] for r in records}) == len(records), "BLOCKED_PROFILE", "duplicate source files")
    return digest(sorted(records, key=lambda r: r["path"]))


def family(device):
    name = device["name"].upper()
    patterns = {"RTX4090": r"(?:NVIDIA )?(?:GEFORCE )?RTX 4090",
                "RTX5090": r"(?:NVIDIA )?(?:GEFORCE )?RTX 5090",
                "H200": r"(?:NVIDIA )?H200(?: NVL| SXM(?:5)?(?: 141GB)?)?",
                "H100": r"(?:NVIDIA )?H100(?: PCIE| SXM(?:5)?| NVL| 80GB HBM3| SXM(?:5)? 80GB| PCIE 80GB)?",
                "B200": r"(?:NVIDIA )?B200(?: SXM(?:6)?| SXM(?:6)? 180GB)?"}
    found = [key for key, pattern in patterns.items() if re.fullmatch(pattern, name)]
    require(len(found) == 1, "UNSUPPORTED_GPU", "unrecognized physical GPU name")
    key = found[0]
    require(device["sm"] == {"RTX4090": 89, "RTX5090": 120, "H200": 90, "H100": 90, "B200": 100}[key],
            "UNSUPPORTED_GPU", "GPU name/SM disagreement")
    return key


def request_family(request):
    checked(request, "request")
    ranks = request["ranks"]
    require(len(ranks) == request["parallel"]["sp"], "UNSUPPORTED_SHAPE", "SP requires every physical rank")
    require(len({r["device"]["uuid"] for r in ranks}) == len(ranks), "UNSUPPORTED_GPU", "duplicate physical GPU")
    require(len({r["device"]["pci_bus_id"] for r in ranks}) == len(ranks), "UNSUPPORTED_GPU", "duplicate PCI bus")
    families = {family(r["device"]) for r in ranks}
    require(len(families) == 1, "UNSUPPORTED_GPU", "mixed GPU families need a separate contract")
    for rank in ranks:
        s = rank["shape"]
        require(s["batch"] == 1 and s["dim"] == 128 and s["heads"] == s["kv_heads"] and not s["causal"]
                and s["dtype"] == "bf16" and s["q"] > 0 and s["kv"] > 0,
                "UNSUPPORTED_SHAPE", "original B01 provider requires B1/D128/BF16/MHA/noncausal")
        require(s["history_tokens"] <= s["kv"] and s["history_tokens"] % s["frame_tokens"] == 0,
                "UNSUPPORTED_SHAPE", "invalid framewise historical KV geometry")
        require(len(s["q_strides"]) == 4 and all(len(s[k]) == (4 if s["layout"] == "BSHD" else 5)
                    for k in ("k_strides", "v_strides")),
                "UNSUPPORTED_SHAPE", "actual Q/K/V strides required for layout")
    return families.pop()


def check_config(config, request):
    checked(config, "config")
    require(config["gate_tiles"] & (config["gate_tiles"] - 1) == 0,
            "UNSUPPORTED_SHAPE", "original provider gate_tiles must be a positive power of two")
    require(config["block_m"] == 64 or config["splits"] == 1, "UNSUPPORTED_SHAPE", "M128 SplitKV is not an original candidate")
    require(all(r["shape"]["heads"] % config["head_group"] == 0 for r in request["ranks"]),
            "UNSUPPORTED_SHAPE", "head group must divide every actual rank head count")
    if config["entrypoint"] == "h200_stock_frontier":
        require(config["layout"] == "BSHD" and config["block_m"] == 128 and config["block_n"] == 64
                and config["splits"] == 1 and config["gate_tiles"] == 4
                and config["producer"] == "framewise_dma" and config["consumer_start"] == "first_frame_ready",
                "BLOCKED_PROFILE", "H200 stock Frontier recipe mismatch")
    else:
        require(config["layout"] == "head_group_major" and config["producer"] == "cpp_dma"
                and config["consumer_start"] == "initialized", "BLOCKED_PROFILE",
                "grouped B01 requires grouped KV, native DMA and initialization dependency")
        if config["entrypoint"] in {"ready", "frontier_opt_v9"}:
            require(all(r["shape"]["heads"] == 12 for r in request["ranks"]),
                    "UNSUPPORTED_SHAPE", "legacy grouped consumer supports H12; use the original autotune provider for other heads")
            require(config["block_m"] == 64 and config["splits"] == 1
                    and config["block_n"] == (32 if config["entrypoint"] == "ready" else 64),
                    "UNSUPPORTED_SHAPE", "legacy grouped consumer tile/split differs from its compiled entrypoint")


def load_catalog(path):
    catalog = checked(read_json(path), "catalog")
    ids = [p["id"] for p in catalog["profiles"]]
    require(len(ids) == len(set(ids)), "BLOCKED_PROFILE", "duplicate profile IDs")
    for r in catalog["recipes"]:
        check_recipe_scope(r)
        for evidence in r["provenance"]["references"]:
            portable(evidence["path"])
    return catalog


def verify_provenance(provenance, root, assets=None):
    for record in provenance["references"]:
        asset_root = (assets or {}).get(record["asset"], root)
        verify_artifact(record, asset_root)


def load_profile(path):
    return checked(read_json(path), "profile")


def recipes_for(catalog, request):
    name = request_family(request)
    fields = ("batch", "q", "kv", "heads", "kv_heads", "dim", "dtype", "causal",
              "history_tokens", "frame_tokens", "layout")
    matches = []
    for recipe in catalog["recipes"]:
        check_recipe_scope(recipe)
        if recipe["family"] != name:
            continue
        if recipe.get("binding_scope", "exact_shape") == "exact_shape":
            if (recipe["shape"]["sp"] != request["parallel"]["sp"] or
                    not all(all(recipe["shape"][k] == rank["shape"][k] for k in fields)
                            for rank in request["ranks"])):
                continue
        else:
            allowed = recipe["kernel_constraints"]["local_heads"]
            if allowed and any(rank["shape"]["heads"] not in allowed for rank in request["ranks"]):
                continue
        try:
            check_config(recipe["config"], request)
        except ProfileError as exc:
            if exc.code == "UNSUPPORTED_SHAPE":
                continue
            raise
        matches.append(deepcopy(recipe))
    return matches


def check_recipe_scope(recipe):
    """A schedule proposal is explicit and belongs to one real native provider."""
    scope = recipe.get("binding_scope", "exact_shape")
    constraints = recipe.get("kernel_constraints")
    require(scope != "schedule" or constraints is not None,
            "BLOCKED_PROFILE", "schedule recipe requires explicit kernel_constraints")
    if constraints is not None:
        provider = {"ready": "legacy_grouped", "frontier_opt_v9": "legacy_grouped",
                    "autotune": "autotune", "split_5090": "split_5090",
                    "h200_stock_frontier": "h200_stock_frontier"}[recipe["config"]["entrypoint"]]
        require(constraints["provider"] == provider, "BLOCKED_PROFILE", "recipe provider/entrypoint mismatch")
        allowed = constraints["local_heads"]
        require(len(set(allowed)) == len(allowed), "BLOCKED_PROFILE", "duplicate local-head constraints")
        if provider == "legacy_grouped":
            require(allowed == [12], "BLOCKED_PROFILE", "legacy H12 kernel cannot declare arbitrary heads")


def verify_binding(profile, request, root, catalog, *, assets=None):
    profile = checked(profile, "profile")
    name = request_family(request)
    require(profile["family"] == name, "UNSUPPORTED_GPU", "physical GPU family differs")
    expected_policy = "autotuned" if name in {"H100", "B200"} else "frozen"
    require(profile["policy"] == expected_policy, "BLOCKED_PROFILE", "frozen device cannot be autotuned")
    require(profile["enabled"] and not profile["disabled_reason"]
            and profile["id"] not in catalog["disabled_profile_ids"]
            and digest(profile["config"]) not in catalog["disabled_config_sha256"],
            "BLOCKED_PROFILE", "profile/configuration manually disabled")
    require(profile["key"]["abi"] == request["abi"] and profile["key"]["b01"] == request["b01"],
            "BLOCKED_ABI", "driver/toolkit/Torch/Python/C++ ABI or B01 source/kernel changed")
    require(profile["key"] == request, "UNSUPPORTED_SHAPE", "physical device, strides, geometry or SP differ")
    check_config(profile["config"], request)
    verify_provenance(profile["provenance"], root, assets)
    if expected_policy == "frozen":
        matches = [r for r in recipes_for(catalog, request) if r["config"] == profile["config"]
                   and r["provenance"] == profile["provenance"]]
        require(len(matches) == 1, "BLOCKED_PROFILE", "no frozen scheduling/source recipe for actual provider; no fallback")
    for record in profile["source_files"]:
        verify_artifact(record, root)
    require(source_digest(profile["source_files"]) == request["b01"]["source_sha256"],
            "BLOCKED_ABI", "B01 source manifest differs")
    verify_artifact(profile["kernel"], root, "BLOCKED_ABI")
    require(profile["kernel"]["sha256"] == request["b01"]["kernel_sha256"], "BLOCKED_ABI", "loaded kernel hash differs")
    return profile


def bind_frozen(catalog, recipe_id, request, source_files, kernel, root, *, assets=None, profile_id=None):
    """Bind an explicitly named frozen recipe to one measured identity; Golden stays absent."""
    matches = [r for r in recipes_for(catalog, request) if r["id"] == recipe_id]
    require(len(matches) == 1, "UNSUPPORTED_SHAPE", "frozen recipe provider does not support the actual shape/SP")
    recipe = matches[0]
    profile = seal({"schema_version": 1, "id": profile_id or recipe_id + "-bound",
                    "family": recipe["family"], "policy": "frozen", "enabled": True, "disabled_reason": "",
                    "key": request, "config": recipe["config"], "source_files": source_files,
                    "kernel": kernel, "provenance": recipe["provenance"], "golden": None})
    return verify_binding(profile, request, root, catalog, assets=assets)


def select_profile(catalog, request, root, *, gpu=False, assets=None):
    """Return an exact binding, or a classified blocker; never select Native/torch."""
    catalog = checked(catalog, "catalog")
    name = request_family(request)
    if name in {"RTX4090", "RTX5090", "H200"}:
        require(bool(recipes_for(catalog, request)), "UNSUPPORTED_SHAPE", "no frozen Mac scheduling recipe supports actual provider/shape")
    candidates = [p for p in catalog["profiles"] if p["family"] == name and p["key"] == request]
    if not candidates:
        same_device = [p for p in catalog["profiles"] if p["family"] == name
                       and p["key"]["ranks"] == request["ranks"] and p["key"]["parallel"] == request["parallel"]]
        require(not same_device, "BLOCKED_ABI", "exact shape exists only for a different ABI/kernel")
        raise ProfileError("BLOCKED_PROFILE", "no exact bound profile; target validation required")
    require(len(candidates) == 1, "BLOCKED_PROFILE", "ambiguous exact profiles; disable/remove stale binding")
    profile = verify_binding(candidates[0], request, root, catalog, assets=assets)
    acceptance = {"request_sha256": digest(request), "config_sha256": digest(profile["config"]),
                  **{stage: {"status": "NOT_RUN", "evidence": None} for stage in
                     ("source_abi", "shape_kernel", "synthetic_golden", "real_qkv_golden", "forward", "overlap")}}
    # Static identity/configuration checks do not prove a loaded GPU kernel,
    # numerical correctness, checkpoint Forward or actual DMA overlap.
    for stage in ("source_abi", "shape_kernel"):
        acceptance[stage]["status"] = "STATIC_PASS"
    if gpu:
        from .runner import verify_golden
        verify_golden(profile, request, root)
        acceptance["real_qkv_golden"] = {"status": "CUDA_GOLDEN_PASS", "evidence": profile["golden"]}
    checked(acceptance, "acceptance")
    return {"status": "CONFIG_AVAILABLE" if gpu else "GPU_GATE_REQUIRED", "profile": profile,
            "validation_level": "CUDA_GOLDEN" if gpu else "STATIC",
            "acceptance": acceptance, "video_e2e": "NOT_RUN", "official_eval": "NOT_RUN"}


def disable_profile(catalog, profile_id, reason):
    require(bool(reason.strip()), "BLOCKED_PROFILE", "operator disable reason required")
    catalog = checked(catalog, "catalog")
    matches = [p for p in catalog["profiles"] if p["id"] == profile_id]
    require(len(matches) == 1, "BLOCKED_PROFILE", "unknown profile ID")
    profile = matches[0]
    profile.update(enabled=False, disabled_reason=reason)
    profile.update(seal(profile))
    if profile_id not in catalog["disabled_profile_ids"]:
        catalog["disabled_profile_ids"].append(profile_id)
    return seal(catalog)
