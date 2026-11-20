# Reproduction evidence and limits

Keep four classes separate: original Table1/2 profiling records; new SOURCE_IDENTICAL_REPLAY executions from recovered research trees; new pruned-source deployments; and earlier compatibility-patched experiments (excluded from this candidate).

The final candidate completed four original real cases on H200 SP1; results and scope are in RELEASE_STATUS.md. These runs use the preserved original timers and external parent controllers. A bare native worker alone does not reproduce the parent NVML monitor/preparation/admission protocol.

Earlier original-tree H200 replay E2E values were379.367952302s (LongLive),88.390334906s (canonical LingBot1.3B),274.218020578s (LingBot14B), and521.086457745s (Solar), with historical video hashes identical. They are not substituted for candidate deployment measurements. LongLive's original-tree replay failed the15% latency tolerance; only the combined pre-diffusion setup increase was localized, with no unmeasured finer cause asserted.

The first candidate LingBot14B run produced a matching full video but failed the original compilation admission:45 compiler execution events in the timed window. Formal E2E remains rejected. A separate original warm-cache preparation and run passed; no compilation time was subtracted. Other earlier failed attempts and raw observations remain evidence, not accepted measurements.

Original5090 LingBot14B generated a complete matching video but its parent ffprobe PATH finalization failed; a zero handled parent exit is not full acceptance and no parent aggregate is fabricated. Original5090 Solar completed generation/decode/eight-rank admission/cleanup at1026.378416239s and matched its original SP8 video. These are original-tree replays, not candidate SP8 passes. The candidate SP8 scope was made optional; no candidate eight-GPU generation was started.

The Table1 efficiency input is the recovered WorldAttention160 WA_001 60s/957-frame six-prompt case; it is not an efficiency average over the VBench-Long quality cohort. The Table2 efficiency case is the recovered WorldAtlas memory-dolly_loop-memory_pool_30s_144 input/480-frame output, separate from ORBIA quality evaluation. No complete quality benchmark was rerun here. Different hardware/cases/profiling histories are not merged into a new independent cohort.

External models, inputs, authorized native code and matching environments must be provisioned separately. Public source/config locks identify content; they do not expose private receipts, grant external redistribution rights or provide an out-of-the-box full-paper numerical reproduction environment.
