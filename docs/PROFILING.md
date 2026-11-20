# Original profiling controllers

The outer CLI accepts `--profile {off,summary}`, default `off`. It does not implement a profiler, change timers, or edit a plan's `gpu_profile` field.

* **off** executes exactly the existing native-worker command and environment. This means no additional parent profiling controller. An original plan that already enables native profiling keeps that setting; this option does **not** promise to disable instrumentation embedded in an immutable scientific plan.
* **summary** selects the backbone's own original external campaign controller, checks its known SHA-256 and prepared metadata, then executes that controller's complete entry point. It never calls `run_one()` in isolation, prepares a campaign, modifies scientific settings, or substitutes another backbone's memory measurement. The original result/admission files are the summary; no new `profile/summary.json` contract is introduced.

## Entry mapping and availability

| Backbone | off (unchanged SP1 worker) | summary controller | Release dispatch status |
|---|---|---|---|
| LongLive-1.3B | `torch.distributed.run ... -m draftkv.backbones.worker --plan PLAN` | `controller_user_repair2_draft_admission_retry.py`, top-level script | Conditionally wired; requires its original fixed interpreter, physical GPU binding, CUDA/FA3 environment and fresh single-row queue |
| LingBot-1.3B | `torch.distributed.run ... worker.py PLAN` | `controller.py`, original `main(g)` | Conditionally wired; requires original tools/environment, physical index/UUID binding and fresh single-row inventory |
| LingBot-14B | `torch.distributed.run ... worker.py PLAN` | `run_campaign.py`, original `main()` | **BLOCKED**: main calls prepare, rewrites scientific plans and launches four rows; fixed old source binding |
| SolarWM-33B | `torch.distributed.run ... worker.py PLAN` | `controller.py --run` | **BLOCKED**: fixed old source/commit/reference binding; recovered controller lacks historical self-hash attestation |

The four full protocols cannot share one controller. SP8 `summary` is also **BLOCKED**: these are original Table 1/2 SP1 controller identities. Existing SP8 worker selection in `off` is unchanged. Blocked requests fail before generation; they never silently fall back to the direct worker or another controller.

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

LongLive needs the original one-row `user_requested_repair2/draft_admission_retry_queue.json`; its fixed interpreter, GPU UUID/index and FA3 override cannot be remapped by this CLI. LingBot-1.3B needs one original-schema `inventory.json` row bound to `run_dir/plan.json`, exact `controller.py`/`worker.py`/`execute.py`/`prepare.py` tool hashes and the original environment module (SHA-256 `52493dc7023a107765aa73ff2ceda934bbc8d91286a61e38dd220e404b676a38`). The original provisioning paths remain constraints. Missing/altered dependencies, an unknown controller, a different source tree, extra rows or a previous attempt block dispatch. Install external dependencies only with appropriate authorization; a SHA check is not a redistribution license.

## Preserve the original scientific measurements

| Backbone | E2E boundary | Table peak memory |
|---|---|---|
| LongLive / LingBot-1.3B | Generation through VAE decode; excludes later video encoding | Parent NVML whole-device sampled generation peak; requested 100 ms cadence; sampled lower bound |
| LingBot-14B | Generation through VAE decode; excludes later video encoding | Maximum per-rank PyTorch allocated peak |
| SolarWM-33B | Generation through independent decode and native video writing | Maximum of sampler/decoder PyTorch allocated peaks, never their sum |

All retain the original synchronized E2E and real diffusion seams, pre-timer extension preparation, compilation admission and raw cold runtime costs. Model loading is outside E2E. Compilation-contaminated timing remains rejected; no time is subtracted. Solar's original first-FA3 witness remains inside E2E. Table FPS is verified output frames / E2E; native `model_only_fps` uses diffusion time and is a different quantity. Memory scopes must not be relabeled or merged.

Original summaries are `gpu_profile.json`, per-rank traces and campaign `result.json`/`status.json`, plus original NVML/memory artifacts for the small backbones. The parent retains original video SHA/ffprobe/attention-route checks where implemented. Inspect these artifacts and formal admission, not merely an exit code: the original LongLive script can record a row failure and exit zero.

## Acceptance scope

CPU tests cover default/off argv and environment equivalence, controller dispatch with synthetic external fixtures, known identity rejection, missing dependencies, old attempts, extra rows, source binding and metadata/source immutability. Fixtures never execute controllers or GPU work and do not replace production hashes. Full native imports, CUDA compilation, NVML operation and end-to-end generation through this **new** `summary` CLI remain **GPU_NOT_TESTED** for all four backbones. The previously verified H200 videos/results remain separate evidence; they were not launched through this new switch. See [release status](RELEASE_STATUS.md).
