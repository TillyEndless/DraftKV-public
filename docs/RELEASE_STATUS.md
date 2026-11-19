# Release status

## Preserved source versions

| Backbone | Deployed public commit | Source-version evidence |
|---|---|---|
| LongLive1.3B | `970a7b887fc8492dafb8b9ac630a635465b18191` | 957 frames; v1 Summary |
| LingBot1.3B | `5fb858a1a7be5e5fc6f48ab5dda116edc099b2c3` | 357 frames; v2 Summary with timing caveat |
| LingBot14B | `ae6275734eff7dfddea2df0d5c679f576394d88e` | 357 frames; v1 Summary; outer harness had missing ffprobe |
| SolarWM33B | `3d88f83c04be34eee42a93ffe9d9553973c59c66` | 357 frames at 16 FPS after decode resume; incomplete Summary |

All 846 retained implementation-body records remain byte-identical to their
selected snapshots (285 distinct contents). Public snapshots match these
commits; provider files match run-plan hashes. Provider revision labels do not
assert that the corresponding Git commit objects are available. Exact pins are
in `verified_runtime_source_map.json`; configurations are in
[REPRODUCTION](REPRODUCTION.md).

Recorded profiling used RTX 4090 for LongLive/Solar and RTX 5090 for LingBot.
Historical SP is 1/8/8/8 respectively. These are separate workloads/environments,
not controlled cross-model timings or four newly reproduced 30-second runs.

## Solar final verification

The first Summary attempt failed because a CUDA compiler was unavailable.
The existing successor completed sampling and hit the original guard
`KV_CONSTRAINT: sampling tensors remain before independent VAE decode`.
The preserved independent decoder-resume path produced the final video without
rerunning sampling. Its SHA256 and ffprobe metadata were checked independently.
All eight sampler ranks report failure at that guard. Summary remains
`INCOMPLETE`, with aggregate E2E/throughput and other fields unavailable.
The wrapper's successful video result is not complete Summary acceptance.
Status: **SOLAR_PROFILE_SUMMARY_UNVERIFIED**.

## Distribution validation boundary

**NEW_PACKAGE_GPU_NOT_TESTED.** Historical GPU source-version records are not
dynamic acceptance of the pruned distribution. CPU tests cover source/config
identity, static plans, native imports, profile guards, invalid CLI rejection
and missing-dependency failures. The `ci/cpu.yml` template builds wheel/sdist
and checks runtime payload agreement without private models or kernels. To
enable it, install it at `.github/workflows/cpu.yml` with workflow-write
authorization. No hosted Actions run is claimed for this release. CPU validation
does not validate
CUDA ABI, SP collectives, decoding or full generation.

Full-paper numerical reproduction and clean cross-model performance
reproduction are not asserted. B01 availability remains deployment dependent.
These boundaries do not change the preserved algorithms or authorize fallback
attention. [INSTALL](INSTALL.md) explains the required external setup.
