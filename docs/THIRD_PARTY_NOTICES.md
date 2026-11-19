# Third-party notices and dependency boundaries

Apache-2.0 covers project-owned DraftKV source and glue that the project owner
is authorized to license. It does not relicense upstream models, kernels,
weights, datasets or evaluators. Existing source notices remain unchanged.

## Bundled shared project glue

`forcing.geometry` and `forcing.budget` provide local shape/frame-equivalent
accounting required by native planners. `worldkv.source` is project integration,
loader and camera-observation glue used by shared native paths. Keeping these
project-authored helpers does not bundle or license the external WorldKV or
DeepForcing algorithms they reference. No independent baseline runner is
supplied. Original generic workers/catalogs retain dormant names where required
for byte identity; independent baseline implementations are excluded.

## External dependencies

| Dependency | Boundary |
|---|---|
| B01 engineering sources and cache | Not bundled; exact authorized archive/manifest/build specification required |
| Modified FlashAttention/native kernel variants | Uncleared bodies/headers omitted; exact original fingerprints required |
| Native LongLive/LingBot/SolarWM backbone bodies | Separately obtained and source-locked |
| PyTorch, FlashAttention, Triton, model libraries | Installed separately under their own licenses; no binaries bundled |
| Checkpoints and conditioning data | Separately obtained under owners' terms |
| VBench-Long, ORBIA and revisit assets | External source locks, licenses and asset permissions apply |

`verified_runtime_source_map.json` has 49 context-specific external records
(19 distinct path/hash pairs). Fingerprints specify identity, not redistribution
permission. Do not publish restricted code merely because the runner can load
it. Omitted exact dependencies must be supplied with appropriate authorization;
otherwise the runtime fails closed. The root license grants no additional rights
to those sources or assets.
