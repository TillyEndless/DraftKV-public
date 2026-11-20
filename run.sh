#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${DRAFTKV_LAUNCH_PYTHON:-python3}" -B "$TASK_ROOT/tools/launch_original.py" "$@"
