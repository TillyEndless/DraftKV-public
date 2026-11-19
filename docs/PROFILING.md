# Profiling

Only `--profile off` and `--profile summary` are supported. Default: `off`.
Original version-specific profiling source is preserved without migration.

## OFF

OFF directly calls the original generation path. Optional diagnostic CUDA
timing events, diagnostic synchronization, profile aggregation and per-layer
profile output are disabled. Correctness-required stream events, Frontier
ordering and buffer lifetimes remain active. CPU contract tests exercise these
guards; OFF does not mean removing correctness synchronization.

## SUMMARY

```bash
./reproduce.sh run --backbone LingBot14B \
  --devices 0,1,2,3,4,5,6,7 --asset-config assets.local.json \
  --workspace outputs/summary --generation-only --profile summary
```

Read `profile/summary.json` and `profile/ranks/` in the chosen workspace.
Depending on version and completeness, observations include generation E2E,
throughput, Fetch+Attention and H2D event envelopes/bytes, receive-buffer
capacity, allocated/reserved GPU peaks, available NVML samples, and
source/checkpoint/kernel provenance. Null/unavailable is not zero. Inspect
`measurement_status`, missing cases and complete rank coverage before using
aggregates.

Generation E2E starts after model load and surrounds the native generation call.
Native decode is included. Solar native writing lies inside that call;
caller-side encoding for other paths lies outside this seam. First-use
compilation and initialization inside the call remain inside the observation.
SUMMARY adds its original fences and peak reset. It does not subtract inferred
compiler costs or make cold runs equivalent to warm runs. Model-specific
conditioning/decode/write scopes differ, so raw cross-model seconds are not a
controlled comparison.

Fetch+Attention and H2D envelopes can overlap and include waits. Their sum is
not exclusive phase accounting. Receive-buffer capacity, allocator peaks and
sampled NVML usage are distinct. None is CPU pinned allocation or process RSS.

## Preserved contract differences

| Context | Limitation |
|---|---|
| LongLive v1 | No diffusion field |
| LingBot14B v1 | No diffusion field |
| LingBot1.3B v2 | Recorded diffusion is null; raw timing was compiler-contaminated |
| Solar v2 | Video exists after independent decode resume; Summary is incomplete |

These are not a uniform paper-reproduction protocol. Values come from runtime
observations, never paper constants. A resumed decoder cannot repair missing
sampler profiling. Solar status: `SOLAR_PROFILE_SUMMARY_UNVERIFIED`.
