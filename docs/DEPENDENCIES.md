# Exact dependency boundary

Every retained implementation file has a source hash in
`release_keep_manifest.json`. Provider fingerprint lists were shortened by
deletion only; retained expected hashes were not changed. These JSON metadata
changes are `MINIMAL_WRAPPER_CHANGE`, with complete before/after differences
retained in the private release audit. Algorithm and profiling source bodies
were not rewritten.

Required upstream/backbone and native files remain pinned by the original
`configs/backbones/*source_lock*.json` and packaged `data/locks/*.json`.
Unauthorized B01 engineering files, modified FA-derived headers, unknown-license
native variants, and private deployment data are not bundled. Obtain authorized
copies separately. A matching filename or upstream repository name is not a hash
match or license grant. Missing exact sources block the affected runtime.

The shared `forcing.geometry`, `forcing.budget` and `worldkv.source` files are
kept where imported by the original preparation/geometry paths. They are not
reimplemented or exposed as independent baseline entry points. Generic original
worker/registry files may retain dormant dispatch branches: deleting parts of
those files would violate byte identity. Independent baseline implementation
modules and their command-line providers are omitted.

Some combined configuration/catalog files remain because the original planner
or integrity checker reads the complete file. For example, the original
`release_source_files` helper names a non-DraftKV policy file in its mandatory
configuration list. That small integrity dependency is retained without a
corresponding baseline algorithm or public run option.

Model environments need their original CUDA/PyTorch/FlashAttention, distributed
SP dependencies, model tokenizer/decoder libraries and ffmpeg/ffprobe. The
preserved native planner checks the recorded source/runtime contracts. CPU-only
CLI/build success does not certify any CUDA kernel or topology.
