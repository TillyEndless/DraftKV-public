"""Internal isolated worker; stdin/stdout are JSON, all adapter logs go to run_dir."""

import json
from pathlib import Path
import sys

from draftkv.registry import Group
from .engine import execute
from .state import write_json


def main():
    request = json.load(sys.stdin)
    row = execute(
        Path(request["root"]),
        Group(**request["group"]),
        request["resolved"],
        request["command"],
        request["argv"],
        report_path=request["report_path"],
        **request["options"],
    )
    write_json(Path(request["report_path"]), row)


if __name__ == "__main__":
    main()
