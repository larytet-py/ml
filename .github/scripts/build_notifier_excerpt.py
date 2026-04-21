#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build notifier excerpt and publish workflow outputs.")
    parser.add_argument("--summary-json", default="notifier_summary.json", help="Path to notifier summary JSON.")
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Path to GitHub Actions output file. Defaults to GITHUB_OUTPUT env var.",
    )
    args = parser.parse_args()

    if not args.github_output:
        raise RuntimeError("Missing --github-output and GITHUB_OUTPUT is not set.")

    summary_path = Path(args.summary_json)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    has_signal = bool(data.get("signals_fired"))

    excerpt = json.dumps(data, indent=2, ensure_ascii=False)
    with open(args.github_output, "a", encoding="utf-8") as fh:
        fh.write(f"has_signal={'true' if has_signal else 'false'}\n")
        fh.write("excerpt<<EOF\n")
        fh.write(excerpt + "\n")
        fh.write("EOF\n")


if __name__ == "__main__":
    main()
