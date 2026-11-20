"""Offline inventory generator, separate from the user-facing CLI."""

import argparse
import json
from pathlib import Path

from .catalog import coverage


def main():
    parser = argparse.ArgumentParser(description="Generate the 26 backbone/algorithm route contracts (no GPU)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(coverage(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
