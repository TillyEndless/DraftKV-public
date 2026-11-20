# DraftKV runtime integration contract (ABI v1)

This contract is owned by CLI/runtime. Backbone, algorithm, dataset, evaluator,
scheduler and audit implementations remain in their owners' directories. No real
backbone, algorithm kernel or official evaluator is substituted by this runtime.
`SUCCEEDED` means the requested orchestration stages succeeded. It does not mean
release acceptance, correct algorithms, valid video decoding or official scores.
Tests use explicit CPU fixtures and establish none of those release claims.

## Registry and default entry points

`Registry.select(benchmark, backbone=None, algorithm=None)` independently filters
both optional fields, case insensitively, using canonical names. The release inventory is defined by [Registry](../registry.py):
Bench1 uses LongLive1.3B; Bench2 uses LingBot1.3B/LingBot14B/SolarWM33B.
Algorithm identities and release targets follow that source inventory. The
[release checklist](../../../docs/release_checklist.md) records the catalogue
size and static publication checks; a campaign may select a smaller subset.
WorldAtlas is not a benchmark. Wan5B/Zing are absent from release selection.
Explicit registry extensions are non-release experiments only.

`AdapterRegistry` contains import-free `AdapterSpec(descriptor, factory, preflight)`
declarations. `register_backbone(name, spec)` and `register_algorithm(name, spec)`
support embedded integrations. The CLI uses these module defaults:

| Identity | Module |
| --- | --- |
| LongLive1.3B | `draftkv.backbones.runtime` |
| LingBot1.3B, LingBot14B | `draftkv.backbones.runtime` |
| each algorithm, including native | `draftkv.algorithms.<canonical algorithm>` |

Every default module exports `describe(config)`, `preflight(context)` and
`create(context)`. Both LingBot sizes share a module; its descriptor selects the
exact `config['backbone']` variant. Descriptor identities must match the requested
canonical backbone/algorithm exactly. Native is an explicit algorithm integration;
it is never installed automatically as a fallback. Missing imports, functions or
kernel assets result in `BLOCKED_INTEGRATION` with traceback and nonzero CLI exit.

Module names can differ. Set `backbone_adapter` or `algorithm_adapter` to an object
with the three explicit `module:callable` references `descriptor`, `factory` and
`preflight`. This supports external model repositories without copying their code
into runtime. Modules must be importable in the selected Python environment.

## Capabilities and model/plugin lifecycle

`describe(config)` returns `draftkv.runtime.Capabilities` or its exact JSON fields:

```python
Capabilities(
    identity=config['backbone'],                 # algorithm descriptor uses algorithm
    input_modes=('t_to_v',),                      # LongLive is pure text; LingBot14 uses t_i_to_v
    protocols=('bench1-worldattention-160-t_to_v',),
    supported_sp=(1, 2, 4),
    min_free_memory_mib={'1': 24000, '2': 16000, '4': 12000},
    streaming_kv=True,
    required_weights=('checkpoint', 'text_encoder'),
    kernels=(),
    gpu_names=(),                                # empty: adapter preflight must validate GPU compatibility
    requires_streaming_kv=False,
)
```

These numbers illustrate the schema, **not verified model memory requirements**.
The backbone declares baseline free memory per rank; the algorithm declares
additional free memory. Runtime checks their sum against current free VRAM.
Declarations must cover each supported SP. Actual hardware/precision/topology and
pair-specific memory/compatibility constraints must also be checked by both
`preflight(context)` functions. They return `{'status': 'READY', ...}` or raise;
returning `BLOCKED_*` blocks the run. Return stable version/ABI/precision metadata
under an optional `fingerprint` object; it is persisted and bound into resume
compatibility. Include exact framework/kernel ABI versions and external model
source Git/content hashes in that stable fingerprint. Other check results are
also saved. SP support must never be guessed from count.
A race after the read-only VRAM probe may still cause OOM: generation must fail.

Every non-native algorithm must declare at least one pinned kernel or source
implementation asset in `kernels`: `{'path': '${KERNEL_FILE}', 'sha256': '<pin>'}`.
Runtime checks nonempty file content and SHA-256 without loading the binary.
`preflight` must validate actual kernel ABI, framework versions and GPU support
using CPU-safe metadata or read-only tools; successful file hashing alone does
not prove kernel functionality. If runtime CUDA execution exposes a missing
capability, raise; never continue with a different algorithm.

`describe` and `preflight`, including module top-level code, must be pure CPU
metadata code: no torch/transformers/triton/CUDA-extension imports, model loading,
weight downloads or CUDA allocation. Runtime guards those imports during
preflight (known CPU Pillow/NumPy decoders are allowed). Kernel imports and model loading belong inside `create`/`attach`/
`generate`, after all preflight gates. The configured checkpoint assets must be used directly; integrations must not
silently fetch replacement weights. CLI workers set Hugging Face/Transformers/
Datasets offline environment flags, preventing implicit model downloads.

The backbone factory returns an object implementing:

- `setup(context)`: prepare/validate the already supplied implementation.
- `generate(context)`: execute the real model and commit every pending case.

The algorithm factory returns an object implementing
`attach(backbone, context)`, which installs the requested algorithm on that
backbone. Runtime calls it before `generate`. Methods return `None` on normal
completion or an explicit successful status object; `False`, exceptions and
`BLOCKED_*` are failures. No import-time registration side effects are required.

The CLI starts a fresh Python worker for every group, including evaluations.
The worker sets `CUDA_VISIBLE_DEVICES` to the selected GPU UUIDs in requested
order, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and `WORLD_SIZE=SP`. `--devices` always means
physical nvidia-smi IDs, never ordinals in an existing visibility mask. A parent
mask may only be narrowed; invalid/MIG/ambiguous masks and excluded devices fail.
Numeric parent mask entries are CUDA ordinals, resolved through the PCI-sorted
inventory only when the parent sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`. A numeric
mask under default/FASTEST_FIRST ordering is rejected because its physical
mapping cannot be established without CUDA initialization. Full UUID masks are
matched directly, and nvidia-smi indices are never assumed to equal CUDA ranks.
`context.state.device_mapping['ranks']` binds physical ID/UUID to local rank.
Local CUDA ordinals are `0..SP-1`; adapters must not use physical IDs with
`torch.cuda.set_device`. The backbone adapter owns spawning its ranks, rendezvous,
SP collectives and cleanup; runtime does not pretend that setting WORLD_SIZE
executes distributed inference. Inherited rank/rendezvous variables are cleared inside the controller scope
and restored afterward; no global RANK/LOCAL_RANK is assigned to the controller. A rank worker receives its own rank explicitly from the adapter.

The embedded `RuntimeEngine` is for CPU orchestration or a dedicated interpreter;
its cwd/environment scopes are process-global and must not run concurrently.
Production callers should use CLI or `execute_isolated` with the normal registry.

## Configuration and frozen dataset hooks

JSON schema version remains 1. Merge order is `defaults`, matching `backbones`
entry, matching `algorithms` entry, exact `groups` override. Objects recursively
merge; arrays replace. `model_config`, `algorithm_config` and `consumer_config`
are preserved; runtime does not translate algorithm knobs. Use `--config` or the `DRAFTKV_CONFIG` environment variable to select the JSON.
Relative paths resolve against the Git root. Undefined environment variables fail. `model_dir`, dataset,
checkpoint and named weight paths accept `${ENV_VAR}` and `~`. No server paths
are embedded in source. Example configuration structure:

```json
{
  "schema_version": 1,
  "defaults": {
    "dataset_hooks": {"setup": "release_data:prepare", "resolve": "release_data:resolve"},
    "evaluator_preflight": "release_metrics:preflight",
    "evaluator": "release_metrics:evaluate",
    "auditor": "release_audit:audit"
  },
  "backbones": {
    "LongLive1.3B": {
      "model_dir": "${LONGLIVE_SOURCE}",
      "checkpoint": {"path": "${LONGLIVE_CHECKPOINT}"},
      "dataset": {"path": "${BENCH1_T2V_MANIFEST}", "sha256": "<trusted frozen SHA-256>"}
    }
  },
  "algorithms": {"draftkv": {"algorithm_config": {}}},
  "groups": []
}
```

Replace pins with actual 64-digit hex strings. Use group overrides for
benchmark-specific data when the same backbone appears in both benchmarks.
`weights` maps additional required weight names to `{path, sha256?}` assets.
Dataset and all configured weights are hashed from actual bytes before a model
factory runs. A release dataset requires an external frozen `sha256` pin. File
hash is ordinary SHA-256; directory hash is SHA-256 of sorted JSON pairs
`[relative_posix_path, file_sha256]`. Directory symlinks/empty directories fail.
`dataset_sha` binds the configured asset; it is a manifest digest when the asset
is a manifest file, and a tree digest when it is a directory. Audit integrations
must distinguish these and supply their own correctly pinned manifest evidence.

Dataset functions take `RuntimeContext`. Setup returns an explicit `READY` or
`SUCCEEDED` object. Resolve must verify original prompts/images/references, the
trusted release identity and ordering, then return:

```python
{
    'status': 'READY',
    'protocol': context.config['protocol'],
    'dataset_sha256': context.state.dataset_sha,
    'case_ids': original_ordered_frozen_ids,
    # Additional JSON fields such as cases/reference paths can be provided.
}
```

Do not echo the required protocol without validating the underlying assets.
`context.dataset` holds the resolver result, including any extra per-case inputs.
The list must contain all frozen cases before runtime filters it. Runtime checks:

| Benchmark/backbone | protocol_id | mode | count | target_frames |
| --- | --- | --- | --- | --- |
| Bench1 LongLive1.3B | `bench1-worldattention-160-t_to_v` | `t_to_v` | 160 | null |
| Bench1 LingBot14B | `bench1-frozen-98-t_i_to_v` | `t_i_to_v` | 98 | null |
| Bench2 both LingBots | `orbia-short-480-357` | `t_to_v` | 480 | 357 |

Bench1 family is `bench1`; Bench2 family is `orbia`. The existing dataset module
contains the strict Bench1 `DatasetResolver` and real runtime wrappers.
Default hooks `draftkv.benchmarks:prepare_dataset`/`resolve_dataset` validate the
frozen source manifests and all actual selected inputs. ORBIA 480/357 uses its
strict asset/index resolver. CLI selection preserves frozen order, filters `--case-ids`
then applies positive `--limit-cases`. A subset is never a full benchmark pass.

## Case artifacts, retries and evaluation provenance

`context.state.case_ids` is the current selection. Iterate
`context.pending_case_ids`, write output inside the unique `context.run_dir`, then
call `context.complete_case(case_id, path)` only after that case finishes. It
requires a nonempty regular file, rejects outside paths/symlinks/duplicate cases
and shared paths, computes SHA-256, and atomically saves `case_results.json`.
The complete benchmark's real output format, frame count, decoding and semantics
must be verified by the official evaluator/auditor. Bytes alone do not prove E2E.

A zero process exit or `generate()` return without every selected case committed
is failure. Runtime rehashes all cases, then writes `generation.json` with the
current run identity, asset hashes and artifacts. `context.generation` holds this
receipt. No old output directory is searched during a new generation.

`--resume RUN_ID` requires one exact group and run/pipeline, with identical code
SHA/source fingerprint, requested config/selection, assets, adapter descriptors,
entry-point code hashes and kernel pins. It creates a **new** Run ID, copies and
rehashes committed cases, and sets `parent_run_id`/per-case `resumed_from`.
It does not modify the old run or trust uncommitted files. Pass the same selection
flags again. Active/preflight-only runs cannot be resumed; failures including
worker crashes can. A regular retry without `--resume` regenerates all cases.

Standalone `evaluate` creates a new run and checks one compatible successful
generation of the exact group/code/config/assets/selection. If several exist,
`--source-run RUN_ID` is required (one selected group, also available with
`evaluate --preflight`); if none exist it fails.
It records `source_run_id`, validates every source hash and never manufactures a
new generation receipt. Failed or incomplete source runs cannot be evaluated as
successful generations. Use resume/pipeline for their partial checkpoints.

Evaluation integrations supply:

- `evaluator_preflight(context)`: validate the actual official protocol/dependency
  pins before generation; return `READY`, or explicit `BLOCKED_*`/raise. It must
  not require outputs that have not been generated yet.
- `evaluate(context)`: score only `context.generation` artifacts; return status
  `COMPLETE`/`SUCCEEDED`/`PASS`/`PASSED`, current `run_id`, and
  `generation_sha256=json_sha256(context.generation)`, plus actual report data.
- `audit(context)`: verify current evidence; return a successful status, current
  `run_id`, the same `generation_sha256`, and
  `evaluation_sha256=json_sha256(context.results['evaluate'])`.

Default evaluator hooks are `draftkv.evaluators:preflight`/`evaluate`/`audit`.
They are not currently installed; the existing `OfficialEvaluator` and audit
classes need CPU-safe wrappers from their owners. Do not set `READY` or `PASS`
just to make these checks pass. Protocol-blocked and dependency-blocked reports
are persisted as failures/blocks, with no scores fabricated by runtime. Runtime
success is not the audit tool's release evidence schema or an official pass.

`pipeline` executes setup/preflight → generate → evaluate → audit for each group.
Entry-point/evaluator protocol preflight precedes model creation; no later stage
runs after a failure. Successful evaluation/audit receipts are recorded only after
their identities and hashes have been verified. Worker nonzero/crash exits are
saved as `FAILED/WorkerProcessError`, including durable partial case checkpoints.

Each run has isolated stdout/stderr, full CLI argv, Git SHA/source fingerprint,
requested and resolved config snapshots/hashes, model/algorithm config, weight/dataset
hashes, device/rank map, stage ledger and timestamps. `inspect` reads manifests
without creating runs. `--dry-run` only resolves declarations: no optional imports,
GPU probes, asset reads, setup or run files. `--preflight` performs real CPU
capability/asset/hardware/protocol checks and records `READY`; it never calls a
model factory, generate, evaluate or audit. CLI exit is 0 for PLANNED/READY/
SUCCEEDED and nonzero for any failed or blocked selected group.

## Pending real integration

All 33 release groups have explicit runtime routing. Native, DraftKV and InfLLM
have real hooks; missing original comparison hooks fail with exact semantics
blockers. No group has current GPU/Forward/video/official acceptance. Required integrations are LongLive1.3B,
LingBot1.3B, LingBot14B; native, DraftKV, ShadowKV, InfiniGen, InfLLM, FreeKV
(original RAFT), WorldKV; frozen Bench1 and official ORBIA wrappers; official
VBench-Long/ORBIA protocol preflight/scoring; and release audit wrapper/evidence.
GPU/SP1/2/4/8, hardware memory/compatibility, Golden, checkpoint forward, 357-frame
video E2E and official benchmark results remain unverified release blockers.


## CLI examples

Offline planning (no integrations/assets/GPU needed):

```sh
PYTHONPATH=src python -m draftkv.cli run --benchmark bench1 --algorithm infllm --devices 0,1 --dry-run
PYTHONPATH=src python -m draftkv.cli pipeline --benchmark bench2 --devices 0,2,3,4 --dry-run
```

After installing the real integrations and supplying the external assets/pins,
set `DRAFTKV_CONFIG` to the operator's portable JSON and use the installed
`draftkv` entry point (or `PYTHONPATH=src python -m draftkv.cli`):

```sh
export DRAFTKV_CONFIG=release-config.json
draftkv setup --benchmark bench1 --devices 0,1,2,3
draftkv run --benchmark bench1 --backbone LongLive1.3B --algorithm draftkv --devices 0
draftkv run --benchmark bench1 --algorithm infllm --devices 0,1
draftkv run --benchmark bench2 --backbone LingBot14B --devices 0,1,2,3
draftkv run --benchmark bench2 --devices 0,1
draftkv evaluate --benchmark bench2 --devices 0,1
draftkv pipeline --benchmark bench1 --devices 0,1,2,3
```

Use `--preflight` to check these prerequisites without model construction;
`evaluate --preflight` also checks an existing compatible generation. Legacy
`adapter.commands` routes remain available only with an explicit `capabilities`
object containing both backbone and algorithm capability declarations. Commands
must be argv arrays, never shell strings; they receive the same UUID/SP mapping.
A run command must write `generation.json` with the current `run_id` and an
`artifacts` list of `{case_id, path, sha256}` files inside that run; runtime ingests
and rehashes it before producing its own canonical receipt. Exit 0 without those
artifacts fails. Legacy plugin factories must migrate to the composed interface.
