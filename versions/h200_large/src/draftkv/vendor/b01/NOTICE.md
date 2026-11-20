# B01 source and license audit

This directory contains an immutable, minimal source snapshot from
`B01_pipeline_autotune_v2_20261018`. It is not a new implementation of attention
or a claim that the entire archive is author-original code.

`source_manifest.json` records the SHA256, size, provenance classification,
authority-manifest match and frozen-4090 comparison for every retained file.
`_source_lock.py` pins that manifest. All 195 snapshot files match the authority's
`MANIFEST_SHA256.txt`. The 185 compile inputs occupy 13,797,379 bytes; ten
additional original build/generator/license files bring the snapshot to
13,814,681 bytes. Literal include closure was traversed from all eleven compilation
roots. The 57 external include names are recorded for the Torch/CUDA/system
compiler environment. No binary, benchmark payload, environment or build cache
is vendored.

## Provenance generations

* Archived FlashAttention v2.8.3 forward math carries Tri Dao's notices. The
  archive's `flash.h` / `flash_fwd_kernel.h` contain the streaming Frontier
  additions; these are not advertised as untouched upstream FlashAttention.
* `build/producer_plan.cpp` implements the later C++ pinned-H2D plan. It is not
  the earlier H200 Python-DMA producer. The archived `streaming_producer.cu`
  includes a distinct SM-resident CUDA-to-CUDA test producer and release-store
  helper, retained because the binding links them. Neither replaces ProducerPlan.
* `ready_*` adds group-major pointers and per-head-group frontiers to FA2.
  `generate_ready_header.py` reproduces its header exactly, without changing
  attention math. It is later engineering work, not initial H200 code.
* `frontier_opt_*` and `generate_frontier_opt.py` add later configurable
  frontier caching/tile/sleep variants. They are clearly separated from g4.
* Original `autotune_*`, `setup_autotune.py` and `generate_autotune_header.py`
  provide actual-local-head grouped dispatch, original tile/SplitKV compute and
  merge choices. Their five P6 imports preserve the authority bytes exactly.
* CUTLASS/CuTe headers are an upstream dependency, with existing notices retained.
* New Python files outside `sources/` are the DraftKV compatibility/contract
  layer. Their behavior is not represented as original B01 engineering code.

The v2 archive reports `410948e` **with a dirty worktree**. This is not a clean
frozen source commit. The independent 4090 g4 archive reports frozen commit
`1a598feded230abd12a489a0ce3a88ae8be95865`; its `.git` was excluded, so this audit
verifies archived content hashes, not the historical checkout's cleanliness.
185/195 vendored files match that archive. The original five frontier additions
are absent from/different to its frozen source set:

* `build/generate_frontier_opt.py`
* `build/setup_frontier_opt.py`
* `fa2_custom/frontier_opt_binding.cpp`
* `fa2_custom/frontier_opt_fwd_kernel.h`
* `fa2_custom/frontier_opt_kernels.cu`

This P6 import adds five more original authority files, also absent from/different
to that frozen set; it does not replace the frozen g3/g4 recipe source identity:

* `build/generate_autotune_header.py`
* `build/setup_autotune.py`
* `fa2_custom/autotune_binding.cpp`
* `fa2_custom/autotune_fwd_kernel.h`
* `fa2_custom/autotune_kernels.cu`

The immutable original files live in deterministic `source_snapshot.tar.gz`,
with `sources/` member paths. Historical reports live in
`historical_evidence.tar.gz`, with `evidence/` member paths. Both archive hashes
and every decoded member hash are verified. `integrity.snapshot_files()` exposes
original member bytes; `integrity.extract_sources()` writes only a new external
build directory. Compression avoids changing upstream PTX/string escape bytes
that the repository scanner would otherwise misidentify as machine paths.

The runtime snapshot is never regenerated in place. Original setup scripts are
provenance material; use the separate `setup_compat.py` entry point. Future
source/ABI patches must be separately named, reviewed and hashed; never modify
`sources/` while continuing to claim the old generation.

## License obligations and clearance

* `sources/third_party/cutlass/LICENSE.txt`: NVIDIA BSD-3-Clause, 2017–2025.
  Preserve copyright, conditions and disclaimer in source and binary distribution
  materials, and obey the non-endorsement condition.
* The archive omitted FlashAttention's LICENSE/AUTHORS. Supplemental upstream
  `licenses/flash-attention-LICENSE` and `licenses/flash-attention-AUTHORS` were
  obtained from the official `v2.8.3` tag; URLs and hashes are in the manifest.
  This BSD-3-Clause license also requires copyright/conditions/disclaimer retention
  and prohibits endorsement. This supplemental retrieval does not prove a grant
  for later B01 modifications.
* No explicit redistribution grant for B01 engineering modifications was located
  in the authority archive. **Public release license clearance is BLOCKED.**
  This audit does not assign an author, infer consent from a filename, or apply
  an invented license to those modifications. Resolve with the rights holder
  before public redistribution, including native binaries.
* New compatibility code's project licensing must be settled with the repository
  license policy; this notice grants no additional rights.
