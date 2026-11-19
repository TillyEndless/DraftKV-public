# DraftKV

DraftKV retrieves historical key/value context for long video generation. A
persistent **Draft Bank** stores compact key summaries. **Draft Grounding**
uses camera geometry when available, and **Draft Refinement** selects relevant
historical context for the current query. **Streaming CG-Attention** transfers
and consumes selected KV through reusable GPU buffers.

This release contains Full DraftKV and its original backbone integrations.
Algorithm, CUDA/C++ and profiling source bodies are preserved from deployed
versions. Independent baselines and Camera-only/Draft-only runners are omitted.

## Supported backbones

| CLI name | Historical SP | Recorded source-version evidence |
|---|---:|---|
| `LongLive1.3B` | 1 | Native-duration generation and v1 Summary |
| `LingBot1.3B` | 8 | Generation and v2 Summary; timing caveat |
| `LingBot14B` | 8 | Generation and v1 Summary |
| `SolarWM33B` | 8 | Generation; `SOLAR_PROFILE_SUMMARY_UNVERIFIED` |

These records identify preserved source versions. The new distribution is
**NEW_PACKAGE_GPU_NOT_TESTED**. See [release status](docs/RELEASE_STATUS.md).

## Installation

Python 3.10 or newer is required for the public interface:

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
./reproduce.sh list
./reproduce.sh plan --backbone LingBot14B
```

Static plans need neither CUDA nor model weights. Native generation needs its
original model environment. Follow [INSTALL](docs/INSTALL.md).

## Model weights and external dependencies

The repository includes the DraftKV research implementation
but is not a fully self-contained, out-of-the-box reproduction
environment. Certain native attention and backbone dependencies
must be obtained separately with appropriate authorization.

Weights, inputs and evaluator assets are not bundled. B01, modified attention
kernels and upstream backbone bodies remain exact, hash-checked external
requirements. Fingerprints do not grant redistribution rights. Read
[dependency setup](docs/INSTALL.md#external-sources-and-assets) and
[third-party notices](docs/THIRD_PARTY_NOTICES.md).

## Quick start and video generation

Configure real paths and SHA256 values using `configs/assets.example.json` and
[INSTALL](docs/INSTALL.md), then run:

```bash
./reproduce.sh preflight --backbone LingBot14B --asset-config assets.local.json
./reproduce.sh run --backbone LingBot14B \
  --devices 0,1,2,3,4,5,6,7 --asset-config assets.local.json \
  --workspace outputs/lingbot14b-off --generation-only --profile off
```

Use one GPU for historical LongLive and eight for historical LingBot/Solar.
Use a new output directory per run. Missing assets or changed source hashes
block execution; there is no substitute baseline or attention implementation.

## Performance profiling

`off` is the default; only `off` and `summary` are supported:

```bash
./reproduce.sh run --backbone LingBot14B \
  --devices 0,1,2,3,4,5,6,7 --asset-config assets.local.json \
  --workspace outputs/lingbot14b-summary --generation-only --profile summary
```

Summary writes supported runtime observations to `profile/summary.json` and
per-rank records. Available fields and timing scopes differ across preserved
versions. This is not automatically the paper's original efficiency protocol.
See [PROFILING](docs/PROFILING.md).

## Evaluation

Interfaces support VBench-Long for LongLive, ORBIA quality for LingBot/Solar,
and revisit evaluation for camera-conditioned backbones. Evaluators and assets
are external and source-locked.

```bash
./reproduce.sh evaluate --backbone LingBot14B --evaluator orbia \
  --devices 0 --asset-config assets.local.json \
  --video-map outputs/lingbot14b-off/video-map.json --workspace outputs/orbia
./reproduce.sh evaluate --backbone LingBot14B --evaluator revisit \
  --devices 0 --asset-config assets.local.json \
  --video-map outputs/lingbot14b-off/video-map.json --workspace outputs/revisit
```

Select `--backbone LongLive1.3B --evaluator vbench-long` for LongLive quality.
Evaluation validates video/source/config bindings and does not generate videos.
See [REPRODUCTION](docs/REPRODUCTION.md) for evaluator configuration.

## Reproduction status

Four scientific configurations and their source fingerprints are included.
CPU tests, static plans and package builds have a CI template in `ci/cpu.yml`. Historical GPU
records do not establish full-paper numerical reproduction or GPU validation
of this distribution. Solar Summary remains unverified. Detailed scope is in
[RELEASE_STATUS](docs/RELEASE_STATUS.md).

## License

[Apache-2.0](LICENSE) covers authorized project-owned code. External code,
weights and datasets retain their own terms. See
[THIRD_PARTY_NOTICES](docs/THIRD_PARTY_NOTICES.md).

## Citation

[CITATION.cff](CITATION.cff) provides software citation metadata. A verified
paper citation will be added when available; no publication or acceptance
status is asserted here.

## Acknowledgements

This implementation builds on the LongLive, LingBot-World and SolarWM model
ecosystems, PyTorch, FlashAttention and Triton, and uses VBench-Long and ORBIA
evaluation interfaces. Their respective licenses and asset terms apply.
