# Verified source-release coverage

The outer CLI now accepts `--profile {off,summary}` (default `off`). This is a CPU-tested dispatch addition, not a new GPU acceptance. Existing four-backbone H200 measurements below are unchanged. All four mappings are CLI_STATIC_VERIFIED with original external dependencies and fresh immutable metadata; 14B/Solar use their audited original single-row APIs with explicit startup path/identity bindings. Complete generation through the new switch is NOT_RUN and is not required to repeat the already completed algorithm/Profile acceptance. See [profiling](PROFILING.md).

The minimum acceptance covers four Full DraftKV configurations, one real original case per backbone, using one H200 per process group (SP1). Separately installed original external dependencies were recorded in private integration commits. Each run completed diffusion, historical attention, decode, a full video, original timing admission and process/resource cleanup. Video hashes match the respective historical references.

| Candidate backbone | Output frames | Admitted E2E s | Frames/E2E | Profile classification |
|---|---:|---:|---:|---|
| LongLive1.3B | 957 | 271.7921 | 3.5211 | PROFILE_MATCH_INITIAL_TOLERANCES_SINGLE_CASE |
| LingBot1.3B | 480 | 92.0233 | 5.2161 | PROFILE_MATCH_INITIAL_TOLERANCES_SINGLE_CASE |
| LingBot14B | 480 | 272.5749 | 1.7610 | PROFILE_MATCH_INITIAL_TOLERANCES_SINGLE_CASE |
| SolarWM33B | 480 | 523.4181 | 0.9170 | PROFILE_MATCH_INITIAL_TOLERANCES_SINGLE_CASE |

Solar is FUNCTIONAL_PASS_WITH_SUPPLEMENTAL_CPU_ADMISSION: its original controller exited with code 1 because the original CPU-only measurement_admission companion was omitted from the private validation copy; that failed parent/status/queue and every raw receipt remain unchanged. The exact historical helper and original aggregate/eight checks/qualification were executed once in a new CPU-only copy, passing without new generation, subprocess launch, timer change or subtraction. Full independent video/source/native/cache/teardown verification passed. This is not a clean automatic parent exit; the failure is disclosed separately. The missing external companion and exact hash are now documented in INSTALL.

LongLive's original decode path records completion/shape and produces the verified byte-identical video; elementwise floating-point tensor finiteness is NOT_RECORDED. A LingBot-specific finite event is not inferred for LongLive.

These are single-case source-release checks, not quality-cohort replication. PROFILE_MATCH_INITIAL_TOLERANCES_SINGLE_CASE means the compared case/protocol meets the initial E2E/memory tolerances; it does not establish historical environment/binary identity. The original first-FA3 CUPTI witness remains in Solar's E2E. Memory scopes differ across backbones and are documented with the original profile evidence. LongLive diffusion is 174.3025 s versus the historical 150.5391 s (+15.79%); the specified 15% timing tolerance applies to E2E.

Retained source/config integrity: 1,216 file memberships, 294 unique SHA-256 contents, 0 changes. The earlier frozen-source clean-clone acceptance covered five CPU integrity/CLI tests and six static original-worker plans. The unified profile CLI now has 18 CPU tests; see PROFILE_CLI_ACCEPTANCE.md for its separate static scope. Native compilation/loading/full generation was checked only in the corresponding original GPU environments, not inferred from CPU plans. Historical complete inherited environments/live Python/ELF maps remain partially unavailable.

The additional original SP8 layouts are retained but **candidate SP8 GPU acceptance is NOT_RUN**. Optional private CPU checks exposed two legacy CLI Git-top-level assumptions in each SP8 layout; those failures were not skipped, patched or called PASS. No SP8 performance claim is made for this candidate. The original-source SP8 historical/replay evidence is a separate class.

Source rights review preserves project Apache, vendor NC/SA and other applicable terms. Official Anemoi `270ddf8` is an exact implementation copy except attribution documentation; the torchvision video derivative retains BSD terms. See THIRD_PARTY_NOTICES.md. Complete B01/native archives, modified native attention binaries and broad backbone/model bodies are not distributed and require separate authorization. A source hash is not a redistribution grant.

This source repository adds no common wheel runtime; wheel/sdist installation/build acceptance is not claimed. Original source trees are used separately. No algorithm/provider/compatibility patch, scientific configuration change, profiling-timer redesign or hot-path source-validation layer was added.
