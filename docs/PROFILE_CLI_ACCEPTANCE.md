# Separate algorithm and CLI acceptance

| Backbone | Original Profile provenance | H200 SP1 algorithm/case acceptance | New summary CLI mapping |
|---|---|---|---|
| LongLive-1.3B | Traced | PASS, 271.7921 s versus 242.8743 s (+11.91%) | CLI_STATIC_VERIFIED: original complete parent |
| LingBot-1.3B | Traced | PASS, 92.0233 s versus 87.5091 s (+5.16%) | CLI_STATIC_VERIFIED: original main(g) |
| LingBot-14B | Traced | PASS, 272.5749 s versus 281.4661 s (-3.16%) | CLI_STATIC_VERIFIED: original run_one(row) |
| SolarWM-33B | Traced | FUNCTIONAL_PASS_WITH_SUPPLEMENTAL_CPU_ADMISSION, 523.4181 s versus 504.8374 s (+3.68%) | CLI_STATIC_VERIFIED: original run_one(row, tools) |

Existing videos, raw profiles and SHA identities are preserved. Solar's earlier parent exit code 1 and subsequent independent CPU qualification remain disclosed; generation and source identity passed. These single-case checks are not full quality-cohort reproduction or historical environment/binary identity.

CLI_STATIC_VERIFIED means CPU verification of entry selection, signatures, source/config/controller hashes, worker commands, metadata bindings and fail-closed behavior with prepared external fixtures. It does not certify that an arbitrary installation has its authorized native/model dependencies. The new CLI is not labeled GPU_DYNAMIC_VERIFIED. Existing algorithm acceptance is not downgraded because full generation through this new switch was not repeated.

The 18 CPU tests include four default/off mappings, summary controller selection, large-backbone signature/binding tests, missing helper/reference/tool refusal, immutable plan/source checks, old-attempt/extra-row refusal and SP8 summary rejection. A private CPU check uses the real archived large-backbone functions with process creation intercepted before any GPU worker. No new sampler, timer, profiler, kernel, producer or source-materialization layer was added.

All 1,216 original source/config memberships (294 unique contents) retain their locked SHA. Exact scientific command mapping and third-party license scope remain unchanged. Required original controller/reference/helper SHA values and startup binding limits are documented in [PROFILING.md](PROFILING.md). Existing dependency and licensing restrictions still apply.
