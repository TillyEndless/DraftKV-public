# SPDX-License-Identifier: Apache-2.0
"""Measurement-only adapter around the exact verified external native worker.

The worker/model/kernel implementation is not copied or replaced. Its old JSONL
performance sink and gpu_profile machinery are disabled in both modes; one
public SUMMARY collector owns all measurements. Correctness/failure checks stay.
"""
import os
from pathlib import Path
from draftkv.runtime import profiling


class MeasurementSink:
    """No per-layer records or serialization in OFF or SUMMARY."""
    def __init__(self, folder, rank): self.handle = self
    def __call__(self, event, **fields):
        # Runtime receipts remain separate. Performance is accumulated at the
        # measured streaming seam, not reconstructed from per-layer log files.
        return None
    def close(self): pass


def install(worker, original_plan, release, source_root):
    original_read = worker.read_json
    original_configure = worker.configure_environment
    selected_plan = Path(original_plan['run_dir'])/'backbone_plan.json'
    def read_plan(path):
        value = original_read(path)
        if Path(path).resolve()==selected_plan.resolve():
            # gpu_profile controls the old instrumentor/monitor, not algorithm
            # or kernel admission. The frozen plan file remains untouched.
            return {**value, 'gpu_profile':False}
        return value
    def configure_environment(plan):
        original_configure(plan)
        # Original environment sanitization discards DRAFTKV_* variables.
        # Restore only these explicitly verified public-runtime bindings.
        os.environ['DRAFTKV_EXTERNAL_RUNTIME_ROOT'] = str(Path(source_root).resolve())
        os.environ['DRAFTKV_PUBLIC_RELEASE_PLAN'] = str(Path(release['workspace'])/'execution-plan.json')
        os.environ.pop('KVRAG_PROFILE_OUTPUT',None)
    worker.read_json = read_plan
    worker.configure_environment = configure_environment
    worker.Trace = MeasurementSink
    worker.run_profiled = profiling.run_profiled
