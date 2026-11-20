# DraftKV

DraftKV research implementation. This source release keeps the original Draft Bank, Draft Grounding, Draft Refinement and Streaming CG-Attention implementations in separate historical source trees. Retained algorithm and configuration files are unchanged.

**Four Full DraftKV backbones passed complete H200 SP1 case acceptance with the preserved original code and separately installed dependencies. See [verified coverage](docs/RELEASE_STATUS.md) for performance differences and scope. External dependencies require separate authorized installation.**

## Supported original paths

| Backbone | SP1 original tree | Additional original topology |
|---|---|---|
| LongLive-1.3B | `versions/longlive` | — |
| LingBot-1.3B | `versions/lingbot13` | — |
| LingBot-14B | `versions/h200_large` | SP8: `versions/sp8_lingbot14_d546` |
| SolarWM-33B | `versions/h200_large` | SP8: `versions/sp8_solar_de790` |

Only Full DraftKV generation is selected by the outer CLI. Shared files keep their original paths, including historical algorithm names. These names do not expose those baselines as supported CLI methods. LongLive and ORBIA remain distinct original integrations.

## Installation and execution

See [installation](docs/INSTALL.md), [original command mapping](docs/ORIGINAL_COMMAND_MAPPING.md), [profiling](docs/PROFILING.md) and [reproduction status](docs/REPRODUCTION.md).

```sh
./run.sh list
./run.sh check
./run.sh plan --backbone longlive1.3b --sp 1 \
  --devices GPU-REPLACE-WITH-ALLOCATED-PHYSICAL-UUID \
  --python /path/to/original/environment/bin/python \
  --plan /path/outside/checkout/approved-plan.json
# Change `plan` to `run` after checking allocation and all external dependencies.
```

Plans are the original worker's execution metadata, prepared separately under the approved operational path mapping. The wrapper does not generate scientific configurations, load a new provider, change kernels or install timing callbacks. External campaign workers are required for the ORBIA rows.

`--profile {off,summary}` defaults to `off`, which preserves the existing worker command and immutable plan settings. `summary` selects a SHA-pinned original external profiling controller. LongLive and LingBot-1.3B have conditional dispatch in their separately provisioned original environments; 14B and Solar remain explicitly blocked because their unchanged controllers cannot safely bind this release. No new-switch GPU test is claimed. See [profiling requirements and measurement scopes](docs/PROFILING.md).

The repository includes the DraftKV research implementation
but is not a fully self-contained, out-of-the-box reproduction
environment. Certain native attention and backbone dependencies
must be obtained separately with appropriate authorization.

## Evaluation

Original evaluation modules remain in each retained tree where required by the audited dependency closure. Complete ORBIA/VBench-Long quality replication still requires the original benchmark inputs, cohorts and metrics. A single generated video validates that case only.

## License

The root Apache-2.0 applies to project code that the project has the right to license. Third-party files keep their own terms; see [third-party notices](docs/THIRD_PARTY_NOTICES.md). Some retained vendor leaves are CC-BY-NC-SA-4.0. Ragged-2D and the torchvision video adapter retain their matched public Apache/BSD attribution and original notices. No model weights, private run receipts, native archives or compiled binaries are included.
