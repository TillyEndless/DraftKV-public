# Subtractive pruning manifest

Each retained source/config file is recorded in `RETAINED_SOURCE_SHA256.json`. This candidate did not modify retained implementation bytes. External-only dependencies are not included; their omission is not proof that inference can run without them.

| Version | Original indexed files | Retained | Omitted |
|---|---:|---:|---:|
| longlive | 505 | 243 | 262 |
| lingbot13 | 505 | 243 | 262 |
| h200_large | 2077 | 239 | 1838 |
| sp8_lingbot14_d546 | 2223 | 244 | 1979 |
| sp8_solar_de790 | 2226 | 247 | 1979 |

The small-version denominator is the audited original src/config/metadata snapshot; the large-version denominator is its entire original Git tree. These denominators are different and must not be called the same source-only cohort. Omitted categories include independent baseline modules, unbound configurations, old fixtures/telemetry, private research materials, broad external model sources, native archives and uncleared derived headers.

Required original shared helpers, existing package paths and vendor notices remain. Two Solar runtime/model JSON dependencies were restored byte-for-byte after the CPU audit found direct fixed-path reads. Detailed filename-by-filename decisions and original SHA values remain in the separate private audit directory.
