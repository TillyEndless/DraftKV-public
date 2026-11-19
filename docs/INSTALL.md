# Installation

## Public package

Use Python 3.10 or newer, with a separate environment for native models:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
./reproduce.sh list
./reproduce.sh plan --backbone LingBot14B --profile off
```

The interface does not install a generic CUDA stack. Recorded profiling used
PyTorch `2.8.0+cu128` and CUDA 12.8; this is provenance, not a guarantee that
arbitrary wheels, GPUs or extensions are interchangeable. Original native
planners verify backbone-specific dependencies, FlashAttention ABI and kernel
fingerprints. Provide ffmpeg and ffprobe in the model environment.

## External sources and assets

1. Select `src/draftkv_minimal/verified/<backbone>/`. Its `public/` runtime and
   `provider/` native-planner subset form one preserved context. Do not mix
   versions.
2. Obtain authorized omitted native/backbone/B01 sources. The
   `external_dependencies` array in `verified_runtime_source_map.json` gives
   exact paths and SHA256 values. Original source/runtime locks remain in each
   context's `provider/configs/backbones/` and `public/src/draftkv/data/locks/`.
   Supply exact authorized files in a separate local deployment.
3. Use compatible original environments and binaries. B01 requires its exact
   source archive and original manifest, compiled cache, and a hash-bound build
   specification containing ABI/source-generation identity. Inspect the
   original requirements with `dependencies --backbone BACKBONE`. This release
   does not provide a public download for restricted B01 code.
4. Obtain model checkpoints and frozen conditioning inputs under their original
   terms. Directory checkpoints require an exact relative-file SHA256 index;
   a path or fabricated combined checksum is insufficient.
5. Copy `configs/assets.example.json` to `assets.local.json`. Replace every
   `/deployment/...` placeholder and `REPLACE_WITH_SHA256`. Set `model_python`,
   authorized deployment `source_root`, checkpoint pins, per-case config paths
   and hashes, and `b01` data. Each case config binds its frozen request,
   source lock, scientific budget and `algorithm_plugin` to that deployment;
   it is not a free-form prompt specification.
6. After pruning changes an inventory, regenerate only that inventory with the
   original provider helper `draftkv.release.audit.release_source_files(root)`.
   Store it in the case config's `algorithm_plugin.files`; its `root` must
   equal `source_root`. Rehash the case config in `assets.local.json`. Keep
   scientific budgets, kernel pins, checkpoint/input hashes and source-lock
   values unchanged. This metadata operation does not certify GPU execution.
7. Run the original validators:

   ```bash
   ./reproduce.sh preflight --backbone LingBot14B --asset-config assets.local.json
   ```

Missing or changed dependencies block execution. Do not disable source checks.
Keep local assets, private dependencies, weights and outputs out of Git.

## CPU validation

```bash
python -m pip install pytest build
python -m pytest -q
python -m build
```

No private data, model downloads or GPU are needed. Passing CPU tests does not
validate native kernels or model generation. See [RELEASE_STATUS](RELEASE_STATUS.md).
