# Original profiling controllers

The outer CLI accepts `--profile {off,summary}`, default `off`. It does not implement a profiler, change timers, or edit a plan's `gpu_profile` field.

* **off** executes exactly the existing native-worker command and environment. This means no additional parent profiling controller. An original plan that already enables native profiling keeps that setting; this option does **not** promise to disable instrumentation embedded in an immutable scientific plan.
* **summary** selects the backbone's own original external campaign controller, checks its known SHA-256 and prepared metadata, then invokes its audited original measurement entry. LongLive and LingBot-1.3B keep their complete parent entries; 14B/Solar use the existing single-row `run_one` API after validating all required prepared metadata and dependencies. The CLI never prepares a campaign, modifies scientific settings, or substitutes another backbone's memory measurement. The original result/admission files are the summary; no new `profile/summary.json` contract is introduced.

## Entry mapping and availability

| Backbone | off (unchanged SP1 worker) | summary controller | Release dispatch status |
|---|---|---|---|
| LongLive-1.3B | `torch.distributed.run ... -m draftkv.backbones.worker --plan PLAN` | `controller_user_repair2_draft_admission_retry.py`, top-level script | **CLI_STATIC_VERIFIED**; requires its original fixed interpreter, physical GPU binding, CUDA/FA3 environment and fresh single-row queue |
| LingBot-1.3B | `torch.distributed.run ... worker.py PLAN` | `controller.py`, original `main(g)` | **CLI_STATIC_VERIFIED**; requires original tools/environment, physical index/UUID binding and fresh single-row inventory |
| LingBot-14B | `torch.distributed.run ... worker.py PLAN` | `run_campaign.py`, original `run_one(row)` | **CLI_STATIC_VERIFIED**: single prepared row, known controller/reference/tool hashes; external environment still required |
| SolarWM-33B | `torch.distributed.run ... worker.py PLAN` | `controller.py`, original `run_one(row, tools)` | **CLI_STATIC_VERIFIED**: single prepared row, known controller/reference/tool/helper hashes; historical self-hash disclosure retained |

These are static CLI statuses, conditional on separately provisioned original dependencies. They do not rerun or downgrade the four completed H200 SP1 acceptances. The four full protocols cannot share one controller. SP8 `summary` is also **BLOCKED**: these are original Table 1/2 SP1 controller identities. Existing SP8 worker selection in `off` is unchanged. Blocked requests fail before generation; they never silently fall back to the direct worker or another controller.

### Recovered controller SHA-256

| Backbone | SHA-256 |
|---|---|
| LongLive-1.3B | `f6a26bc19ab26e57a112ea62f0cb0a5df49e79391336f1a64cdcf800d8075dd6` |
| LingBot-1.3B | `16fd7e0f51f926178d960b64fb448c52591f5977e2cddde1b677081e75df69e5` |
| LingBot-14B | `112d5a5c2920009c81fc4ec6d7f1fe051df98bdb74e24c14f94fa51072de3477` |
| SolarWM-33B | `fe94e35d08106aed71f63f6053b9ee76f56d80d693ffd64dae5c105774ca9ce6` |

These are the recovered original campaign scripts associated with the Table 1/2 receipts, not the later public-profile staging implementation. The 14B command receipt pins its controller. Solar's digest identifies recovered bytes, not a historically recorded controller digest. Exact original **worker** argv/cwd, receipts and file hashes are retained in the separate private provenance audit. Complete historical parent shell commands and inherited environments were not serialized; source-derived script interfaces are not presented as shell receipts. Controller scripts and private deployment paths are not distributed here.

## Usage and required external preparation

Add `--profile off` to any existing [original worker command](ORIGINAL_COMMAND_MAPPING.md), or omit it. For example:

```sh
./run.sh plan --backbone longlive1.3b --sp 1 --profile off \
  --devices GPU-REPLACE-WITH-ALLOCATED-PHYSICAL-UUID \
  --python /path/to/original/environment/bin/python \
  --plan /path/outside/checkout/approved-plan.json
```

For a separately provisioned original LingBot-1.3B controller:

```sh
./run.sh plan --backbone lingbot1.3b --sp 1 --profile summary \
  --controller /path/to/original/campaign/controller.py \
  --controller-device-index 1 \
  --devices GPU-REPLACE-WITH-ALLOCATED-PHYSICAL-UUID \
  --python /path/to/original/environment/bin/python \
  --worker /path/to/original/campaign/worker.py \
  --worker-sha256 33ea5645ce3dfc408b7da8fce6b12bb5304509858dd1e4ee33da69068f693684 \
  --plan /path/outside/checkout/fresh-run/plan.json
```

Use `run` instead of `plan` only with an exclusive allocation and all original dependencies installed. `plan` is CPU-only: it does not import or execute the controller or query the GPU. `run` additionally verifies the original physical index resolves to the requested UUID before executing the parent. The original controller keeps its own resource lock, idle checks, environment and validation.

The external campaign's existing `source` link must resolve to the selected release `versions/` tree. Original scientific/source identity checks still apply. Execution metadata must bind the current clean release commit and an unused output directory. The CLI creates no metadata, directories or links.

LongLive needs the original one-row `user_requested_repair2/draft_admission_retry_queue.json`; its fixed interpreter, GPU UUID/index and FA3 override cannot be remapped by this CLI. LingBot-1.3B needs one original-schema `inventory.json` row bound to `run_dir/plan.json`, exact `controller.py`/`worker.py`/`execute.py`/`prepare.py` tool hashes and the original environment module (SHA-256 `52493dc7023a107765aa73ff2ceda934bbc8d91286a61e38dd220e404b676a38`). The original provisioning paths remain constraints. Missing/altered dependencies, an unknown controller, a different source tree, extra rows or a previous attempt block dispatch. `CLI_STATIC_VERIFIED` does not certify that external runtime dependencies exist on a reader's machine. Install external dependencies only with appropriate authorization; a SHA check is not a redistribution license.

## Large-backbone single-row mapping

For 14B/Solar, supply the existing original-schema `selected_row.json` and one-row `inventory.json` in the external campaign. All plan/scientific fields remain unchanged. No files are created by `plan`.

```sh
./run.sh plan --backbone lingbot14b --sp 1 --profile summary \
  --controller /path/to/campaign/tools/run_campaign.py \
  --controller-reference /path/to/original/serial-reference \
  --devices GPU-REPLACE-WITH-ALLOCATED-PHYSICAL-UUID \
  --python /path/to/original/environment/bin/python \
  --worker /path/to/campaign/tools/worker.py \
  --original-worker /path/to/archived/original/worker.py \
  --worker-sha256 SHA256-OF-APPROVED-PATH-MAPPED-WORKER \
  --plan /path/to/campaign/runs/selected-original-row/plan.json
```

Solar selects `--backbone solarwm33b` and `/path/to/campaign/controller.py`; its worker remains `campaign/tools/worker.py`. The CLI audits the already approved single source-root literal difference in that worker against the original worker. It never writes or transforms source.

The selector imports the SHA-pinned original controller without running `main`/`prepare`, binds only startup `B`, `S`, `COMMIT` and (14B only) `REF`, and calls its unmodified original `run_one`. These are explicit path/current-source-identity bindings, not edits to function bodies or profiling logic. Python remains the controller's original interpreter; physical GPU indices remain 0 for 14B and 1 for Solar. Cache expressions, compile flags and the original environment builder are untouched; missing original cache paths block execution.

Audit of omitted batch orchestration: 14B's prepare copies weight metadata, creates directories and generates four plans; the selector instead requires the already prepared receipt/directories/one-row plan. Source/Git/device checks, worker environment, worker execution, compilation admission and reference/video checks remain inside original `run_one`. Solar's batch main supplies inventory selection and idle waiting; the selector verifies one non-reused row/tools, and `run_one` retains the source guard, idle assertion, environment, complete worker, aggregate/correctness checks and original `qualify`. The original idle assertion now rejects an occupied GPU rather than entering the batch wait loop. Batch queue bookkeeping is not recreated; errors propagate and original attempt logs/results are retained.

Reference `result.json` identities, recovered read-only from the original serial runs and checked against each original streaming video's SHA:

| Backbone | Reference SHA-256 |
|---|---|
| LingBot-14B | `719a67e0b27dd105024fc48d02b6d8271d3d89c0c36db1b33f768e06acf3f905` |
| SolarWM-33B | `9cccb355fa3013d796ed4f40fe683ec2c247f51bd8c4142fb3b1e03094a81888` |

These pin recovered reference contents, not previously unrecorded historical self-hashes. Solar additionally requires the unchanged external `measurement_admission.py`, SHA-256 `ffc75d45d404be8dcaa3bf893060fc5a17015e9ca3da57ae688e2a23ce611c19`, before dispatch. This prevents the missing-helper failure seen in its prior parent run.

## Preserve the original scientific measurements

| Backbone | E2E boundary | Table peak memory |
|---|---|---|
| LongLive / LingBot-1.3B | Generation through VAE decode; excludes later video encoding | Parent NVML whole-device sampled generation peak; requested 100 ms cadence; sampled lower bound |
| LingBot-14B | Generation through VAE decode; excludes later video encoding | Maximum per-rank PyTorch allocated peak |
| SolarWM-33B | Generation through independent decode and native video writing | Maximum of sampler/decoder PyTorch allocated peaks, never their sum |

All retain the original synchronized E2E and real diffusion seams, pre-timer extension preparation, compilation admission and raw cold runtime costs. Model loading is outside E2E. Compilation-contaminated timing remains rejected; no time is subtracted. Solar's original first-FA3 witness remains inside E2E. Table FPS is verified output frames / E2E; native `model_only_fps` uses diffusion time and is a different quantity. Memory scopes must not be relabeled or merged.

Original summaries are `gpu_profile.json`, per-rank traces and campaign `result.json`/`status.json`, plus original NVML/memory artifacts for the small backbones. The parent retains original video SHA/ffprobe/attention-route checks where implemented. Inspect these artifacts and formal admission, not merely an exit code: the original LongLive script can record a row failure and exit zero.

## Acceptance scope

Original algorithm/Profile acceptance and CLI acceptance are separate. The four completed H200 SP1 cases remain accepted, with the existing Solar supplemental-CPU-admission limitation. Adding this option does not require repeating those complete generations.

The 18 CPU tests cover default/off argv and environment equivalence, all four controller selections, 14B/Solar single-row signatures/startup bindings, known identity rejection, missing dependencies, old attempts, extra rows, source binding and metadata/source immutability. Public tests use synthetic external fixtures and never run CUDA. A separate private CPU contract check also reaches the actual archived 14B/Solar `run_one` functions and stops at mocked `subprocess.Popen`, before any GPU process is spawned. Source/controller/reference identities are independently checked against archived evidence.

The verified mapping is **CLI_STATIC_VERIFIED**. Execution of complete GPU profiling through the new CLI is **NOT_RUN**, a separate interface-evidence field rather than a failed or pending revalidation of the original algorithm. No new full GPU generation is required by this CLI-only release acceptance. See [CLI acceptance](PROFILE_CLI_ACCEPTANCE.md) and [existing SP1 results](RELEASE_STATUS.md).
