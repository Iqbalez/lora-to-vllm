"""nimbus-llm CLI: dataset | train | evaluate | quantize | serve | benchmark."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nimbus-llm")
    parser.add_argument(
        "command",
        choices=["dataset", "train", "evaluate", "quantize", "serve", "benchmark"],
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]

    if args.command == "dataset":
        from nimbus_llm.dataset import main as cmd
    elif args.command == "train":
        from nimbus_llm.train import main as cmd
    elif args.command == "evaluate":
        from nimbus_llm.evaluate import main as cmd
    elif args.command == "quantize":
        from nimbus_llm.quantize import main as cmd
    elif args.command == "serve":
        from nimbus_llm.serve import main as cmd
    else:
        from nimbus_llm.benchmark import main as cmd
    return cmd(rest)


if __name__ == "__main__":
    sys.exit(main())
