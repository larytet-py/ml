#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def _compact_config(config: dict) -> dict:
    compact = {
        "config_index": config.get("config_index"),
        "strategy_name": config.get("strategy_name"),
        "strategy_note": config.get("strategy_note"),
        "symbol": config.get("symbol"),
        "configured_side": config.get("configured_side"),
        "date": config.get("date"),
        "close": config.get("close"),
        "roc_lookback": config.get("roc_lookback"),
        "vol_window": config.get("vol_window"),
        "pricing_vol_annualized": config.get("pricing_vol_annualized"),
        "fired_signals": config.get("fired_signals", []),
    }

    selected_sides = config.get("selected_sides") or []
    if "call" in selected_sides:
        compact["call"] = {
            "trigger": config.get("call_trigger"),
            "roc": config.get("roc"),
            "roc_threshold": config.get("call_roc_threshold"),
            "upside_vol_annualized": config.get("upside_vol_annualized"),
            "upside_vol_threshold": config.get("upside_vol_threshold"),
        }
    if "put" in selected_sides:
        compact["put"] = {
            "trigger": config.get("put_trigger"),
            "roc": config.get("roc"),
            "roc_threshold": config.get("put_roc_threshold"),
            "downside_vol_annualized": config.get("downside_vol_annualized"),
            "downside_vol_threshold": config.get("downside_vol_threshold"),
        }
    return compact


def _build_compact_summary(data: dict) -> dict:
    return {
        "generated_at_utc": data.get("generated_at_utc"),
        "config_count": data.get("config_count"),
        "signals_fired": bool(data.get("signals_fired")),
        "signal_count": data.get("signal_count"),
        "configs": [_compact_config(cfg) for cfg in data.get("configs", [])],
    }


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
    compact_data = _build_compact_summary(data)

    excerpt = json.dumps(compact_data, indent=2, ensure_ascii=False)
    with open(args.github_output, "a", encoding="utf-8") as fh:
        fh.write(f"has_signal={'true' if has_signal else 'false'}\n")
        fh.write("excerpt<<EOF\n")
        fh.write(excerpt + "\n")
        fh.write("EOF\n")


if __name__ == "__main__":
    main()
