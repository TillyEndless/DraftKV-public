# Reproduction

## Scientific configurations

| Backbone | S/L/R/C | Protected history P | Grounding Top-M | Historical SP |
|---|---|---:|---:|---:|
| LongLive1.3B | 3/0/6/3 | 2 | N/A | 1 |
| LingBot1.3B | 3/0/11/4 | 3 | 3 | 8 |
| LingBot14B | 3/0/11/4 | 3 | 3 | 8 |
| SolarWM33B | 0/0/25/5 | 5 | 7 | 8 |

These generation budgets are distinct from streaming receive-slot/tile
parameters. LongLive uses the text-only path; camera-conditioned backbones
include Grounding. Preserve checkpoint, inputs, seed, precision, source/kernel
identity, positions and selection policy when comparing executions. Do not
adjust budgets to target a paper latency.

```bash
./reproduce.sh plan --backbone LongLive1.3B --profile off
./reproduce.sh plan --backbone LingBot1.3B --profile summary
./reproduce.sh plan --backbone LingBot14B --profile off
./reproduce.sh plan --backbone SolarWM33B --profile summary
```

Historical topology is the default. The unchanged LingBot1.3B catalog also
records an earlier SP1 validation, separate from its SP8 profiling record.
It does not validate a new topology for every model. TP is not independently
attested by the available source-version receipts.

## Generation

Configure authorized sources and frozen cases following [INSTALL](INSTALL.md).
Use a new output directory per invocation:

```bash
./reproduce.sh run --backbone LongLive1.3B --devices 0 \
  --asset-config assets.local.json --workspace outputs/longlive \
  --generation-only --profile off
```

Historical LingBot/Solar configurations use eight explicitly selected devices.
Workers validate frozen input contracts and write hash-bound `video-map.json`.
`--generation-only` skips evaluation; without it, the configured quality
evaluator is also required. No independent baseline/ablation runner is supplied.

## Evaluation

Provide source-locked evaluator adapters, environments and weight indices in
`assets.local.json` under `evaluators`. The original 29-file evaluation adapter
lock is retained in full. Required settings include:

- VBench-Long: `adapter_root`, `python`, evaluator `config`, `weight_index`.
- ORBIA: additionally exact upstream `source`, config/hash, dataset lock/hash,
  dataset specification, interface and recorded run/checkpoint/condition
  metadata required by the original bridge.
- Revisit: exact upstream `source`, `python`, `weights`, `adapter_root`, and
  `weight_index`; the original interface invokes the external reprojection script.

Each weight-index entry has a local path and SHA256. Evaluation verifies video
hashes and scientific source/config identity. Arbitrary MP4 files do not satisfy
that binding. A separate evaluation workspace can consume the matching map:

```bash
./reproduce.sh evaluate --backbone LingBot14B --evaluator orbia \
  --devices 0 --asset-config assets.local.json \
  --video-map outputs/lingbot14b/video-map.json --workspace outputs/orbia
./reproduce.sh evaluate --backbone LingBot14B --evaluator revisit \
  --devices 0 --asset-config assets.local.json \
  --video-map outputs/lingbot14b/video-map.json --workspace outputs/revisit
```

Select `--evaluator vbench-long` for LongLive. Revisit never launches generation;
an evaluator may still need GPU resources. These interfaces do not imply that
this release reran every paper cohort. See [RELEASE_STATUS](RELEASE_STATUS.md)
and [PROFILING](PROFILING.md).
