# Original command mapping

The outer `run.sh` selects an original tree and executes its original raw-plan worker using `torch.distributed.run`. It never calls a new config generator. Scientific plugin parameters, effective budget, historical policy, kernel identity and original profile policy must match the SHA identity in `ORIGINAL_COMMAND_MAPPING.json`.

| Row | Original entry |
|---|---|
| LongLive SP1 | `python -u -m torch.distributed.run --standalone --nproc_per_node=1 -m draftkv.backbones.worker --plan PLAN` |
| LingBot-1.3B SP1 | Same original torchrun launcher, original external campaign `worker.py PLAN` |
| LingBot-14B / Solar SP1 | Same original torchrun launcher, original external campaign `tools/worker.py PLAN` with audited startup-only source-path mapping |
| LingBot-14B / Solar SP8 | Original torchrun `--nproc_per_node=8`, original external `worker.py PLAN`; existing external `source` link targets selected original version |

The ORBIA worker and companion original files must be obtained separately. The JSON records their verified original worker SHA. H200 large-model startup path mapping is verified against an external copy of the original worker, allowing exactly one source-root literal difference. No transformed code is generated at runtime. Original Solar FA3 integration remains in its external tools; using a bare module instead would change that execution path.

Execution metadata must record the current candidate parent Git commit, retained adapter/plugin inventory, fresh output paths and actual inputs/dependencies. It must live outside the source Git tree. The root CLI validates source/config hashes once before exec; no per-layer or per-tile integrity callback is added.

Native-worker invocation does not replace the original outer campaign's parent memory monitor, preparation or performance admission. The default/explicit `--profile off` retains exactly the worker mapping above, including unchanged original plan flags. `--profile summary` dispatches only to a SHA-pinned, unchanged original controller when its full entry is safe to reuse; see [per-backbone mapping and blockers](PROFILING.md). No controller path literals are rewritten, and no preparation routine is invoked to create/modify plans. The private audit holds exact historical command records and their SHA values; private deployment paths are not copied into this source tree.
