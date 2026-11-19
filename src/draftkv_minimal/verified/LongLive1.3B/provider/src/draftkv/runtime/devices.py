"""Physical nvidia-smi IDs -> UUID visibility -> local ranks; no CUDA imports."""

from dataclasses import asdict, dataclass
import csv
import io
import os
import re
import subprocess

from .config import ConfigurationError, parse_devices


@dataclass(frozen=True)
class Device:
    physical_id: int
    uuid: str
    name: str
    total_memory_mib: int
    free_memory_mib: int
    pci_bus_id: str | None = None


def discover_devices() -> tuple[Device, ...]:
    """Read-only hardware probe. Never initializes torch, allocates VRAM or kills jobs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.free,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        rows = tuple(
            Device(
                int(r[0]),
                r[1].strip(),
                r[2].strip(),
                int(r[3]),
                int(r[4]),
                r[5].strip(),
            )
            for r in csv.reader(io.StringIO(result.stdout), skipinitialspace=True)
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise ConfigurationError(
            f"GPU inventory unavailable (nvidia-smi): {exc}"
        ) from exc
    if (
        not rows
        or len({r.physical_id for r in rows}) != len(rows)
        or len({r.uuid for r in rows}) != len(rows)
    ):
        raise ConfigurationError("GPU inventory is empty or ambiguous")
    return rows


def device_mapping(selected, inventory=None, inherited=None) -> dict:
    selected = parse_devices(",".join(map(str, selected)))
    rows = discover_devices() if inventory is None else inventory
    by_id = {r.physical_id: r for r in rows}
    if len(by_id) != len(rows) or len({r.uuid for r in rows}) != len(rows):
        raise ConfigurationError("GPU inventory identities are ambiguous")
    if any(i not in by_id for i in selected):
        raise ConfigurationError(f"physical GPU absent from inventory: {selected}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if inherited is None else inherited
    if visible is not None:
        allowed = set()
        if len(visible.split(",")) != len(set(visible.split(","))):
            raise ConfigurationError(
                "inherited CUDA_VISIBLE_DEVICES contains duplicates"
            )
        for item in visible.split(","):
            if item.isdecimal() and len(item) < 11:
                if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                    raise ConfigurationError(
                        "numeric inherited CUDA_VISIBLE_DEVICES requires CUDA_DEVICE_ORDER=PCI_BUS_ID or a full UUID mask"
                    )

                def pci_key(row):
                    match = re.fullmatch(
                        r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])",
                        row.pci_bus_id or "",
                    )
                    if match is None:
                        raise ConfigurationError(
                            "numeric inherited mask requires a verified PCI bus inventory"
                        )
                    return tuple(int(v, 16) for v in match.groups())

                pci_order = sorted(rows, key=pci_key)
                if len({pci_key(row) for row in rows}) != len(rows) or int(item) >= len(
                    pci_order
                ):
                    raise ConfigurationError(
                        "numeric inherited mask has no unambiguous PCI ordinal"
                    )
                physical_id = pci_order[int(item)].physical_id
            else:
                matches = [r.physical_id for r in rows if r.uuid == item]
                if len(matches) != 1:
                    raise ConfigurationError(
                        "CUDA_VISIBLE_DEVICES is empty, invalid, MIG-based or ambiguous; explicit physical mapping required"
                    )
                physical_id = matches[0]
            if physical_id in allowed:
                raise ConfigurationError(
                    "inherited CUDA_VISIBLE_DEVICES aliases the same GPU twice"
                )
            allowed.add(physical_id)
        if not set(selected) <= allowed:
            raise ConfigurationError(
                "--devices requests physical GPUs excluded by inherited CUDA_VISIBLE_DEVICES"
            )
    chosen = [by_id[i] for i in selected]
    if any(
        not r.uuid.startswith("GPU-") or not 0 < r.free_memory_mib <= r.total_memory_mib
        for r in chosen
    ):
        raise ConfigurationError(
            "selected GPU inventory has invalid UUID or memory values"
        )
    return {
        "inherited_cuda_visible_devices": visible,
        "inherited_cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "inherited_distributed_environment": {
            k: v
            for k, v in os.environ.items()
            if k in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        },
        "cuda_visible_devices": ",".join(r.uuid for r in chosen),
        "cuda_device_order": "PCI_BUS_ID",
        "world_size": len(chosen),
        "ranks": [
            {"local_rank": rank, "rank": rank, **asdict(row)}
            for rank, row in enumerate(chosen)
        ],
    }


def validate_hardware(capabilities, mapping, *, enforce_memory=True):
    sp = mapping["world_size"]
    for caps in capabilities:
        if sp not in caps.supported_sp:
            raise ConfigurationError(f"{caps.identity} does not support SP{sp}")
        minimum = caps.min_free_memory_mib.get(str(sp))
        if enforce_memory and minimum is None:
            raise ConfigurationError(
                f"{caps.identity} has no memory requirement for SP{sp}"
            )
        for row in mapping["ranks"]:
            if caps.gpu_names and row["name"] not in caps.gpu_names:
                raise ConfigurationError(
                    f"{caps.identity} does not support GPU {row['name']}"
                )
            if enforce_memory and row["free_memory_mib"] < minimum:
                raise ConfigurationError(
                    f"GPU {row['physical_id']} has insufficient free memory for {caps.identity}/SP{sp}: {row['free_memory_mib']} < {minimum} MiB"
                )
    if enforce_memory:
        required = sum(c.min_free_memory_mib[str(sp)] for c in capabilities)
        if any(r["free_memory_mib"] < required for r in mapping["ranks"]):
            raise ConfigurationError(
                f"insufficient free memory for combined backbone/algorithm: need {required} MiB per rank"
            )
