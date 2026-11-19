"""Release targets are routing declarations, never evidence of algorithm support.

No algorithm implementation or checkpoint is bundled by these declarations.
Additional groups can be supplied by config, and are always non-release targets.
"""

from dataclasses import dataclass
import re
from typing import Iterable


class RegistryError(ValueError):
    pass


def canonical_method(value: str) -> str:
    value = value.lower()
    return {"worldtrace-lm": "worldtrace-landmark"}.get(value, value)


def algorithm_package(value: str) -> str:
    return canonical_method(value).replace("-", "_")


BENCH1_DATASETS = {"LongLive1.3B": "bench1-worldattention-160-t2v"}


@dataclass(frozen=True)
class DatasetProtocol:
    """Release identity requirements; these declarations do not authenticate data."""

    protocol_id: str
    dataset_family: str
    input_mode: str
    case_count: int
    target_frames: int | None = None

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def safe_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise RegistryError(f"{label} must be a safe, nonempty name: {value!r}")
    if ".." in value:
        raise RegistryError(f"{label} cannot contain '..': {value!r}")
    return value


@dataclass(frozen=True)
class Group:
    benchmark: str
    backbone: str
    algorithm: str
    release_target: bool = False

    def __post_init__(self):
        if self.benchmark.lower() == "bench1" and self.backbone.lower() == "lingbot14b":
            raise RegistryError("Bench1 LingBot14B and its T+I benchmark have been retired")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.benchmark, self.backbone, self.algorithm)

    def to_dict(self) -> dict:
        return dict(
            benchmark=self.benchmark,
            backbone=self.backbone,
            algorithm=self.algorithm,
            release_target=self.release_target,
        )

    @property
    def protocol(self) -> DatasetProtocol:
        if self.benchmark == "bench2":
            return DatasetProtocol("orbia-short-480-357", "orbia", "t_i_camera_to_v", 480, 357)
        return DatasetProtocol("bench1-worldattention-160-t_to_v", "bench1", "t_to_v", 160)

    @property
    def dataset_id(self):
        return BENCH1_DATASETS[self.backbone] if self.benchmark == "bench1" else "orbia-short-480-357"


class Registry:
    def __init__(self, extensions: Iterable[dict] = ()):
        self._groups: dict[tuple[str, str, str], Group] = {}
        specs = (
            (
                "bench1",
                ("LongLive1.3B",),
                ("native", "draftkv", "shadowkv", "infinigen", "infllm", "freekv", "deep-forcing", "future-forcing", "worldtrace-landmark"),
            ),
            (
                "bench2",
                ("LingBot1.3B", "LingBot14B", "SolarWM33B"),
                (
                    "native",
                    "draftkv",
                    "shadowkv",
                    "infinigen",
                    "infllm",
                    "freekv",
                    "worldkv",
                    "worldtrace-landmark",
                    "deep-forcing",
                    "future-forcing",
                ),
            ),
        )
        for benchmark, backbones, algorithms in specs:
            for backbone in backbones:
                for algorithm in algorithms:
                    group = Group(benchmark, backbone, algorithm, True)
                    self._groups[group.key] = group
        for entry in extensions:
            if not isinstance(entry, dict):
                raise RegistryError("each registry extension must be an object")
            try:
                benchmark = self.benchmark(entry["benchmark"])
                backbone = safe_name(entry["backbone"], "backbone")
                algorithm = safe_name(entry["algorithm"], "algorithm").lower()
            except KeyError as exc:
                raise RegistryError(
                    f"registry extension missing {exc.args[0]}"
                ) from exc
            key = (benchmark, backbone, algorithm)
            for existing in self._groups.values():
                if (
                    existing.benchmark == benchmark
                    and existing.backbone.lower() == backbone.lower()
                ):
                    backbone = existing.backbone
                    key = (benchmark, backbone, algorithm)
                    break
            if any(
                tuple(s.lower() for s in existing) == tuple(s.lower() for s in key)
                for existing in self._groups
            ):
                raise RegistryError(f"duplicate registry group: {key}")
            if entry.get("release_target", False) is not False:
                raise RegistryError("extensions cannot declare release_target=true")
            self._groups[key] = Group(*key)

    @staticmethod
    def benchmark(value: str) -> str:
        if not isinstance(value, str) or value.lower() not in ("bench1", "bench2"):
            raise RegistryError(f"unknown benchmark {value!r}; choose bench1 or bench2")
        return value.lower()

    @property
    def groups(self) -> tuple[Group, ...]:
        return tuple(self._groups.values())

    def select(
        self, benchmark: str, backbone: str | None = None, algorithm: str | None = None
    ) -> tuple[Group, ...]:
        bench = self.benchmark(benchmark)
        if algorithm is not None:
            algorithm = canonical_method(algorithm)
        for label, value in (("backbone", backbone), ("algorithm", algorithm)):
            if value is not None:
                safe_name(value, label)
        groups = tuple(
            g
            for g in self.groups
            if g.benchmark == bench
            and (backbone is None or g.backbone.lower() == backbone.lower())
            and (algorithm is None or g.algorithm.lower() == algorithm.lower())
        )
        if not groups:
            raise RegistryError(
                f"no registered group for {bench}, backbone={backbone!r}, "
                f"algorithm={algorithm!r}"
            )
        return groups
