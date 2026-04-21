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

    lines = [
        "Structured signal summary",
        f"generated_at_utc={data.get('generated_at_utc')}",
        f"config_count={data.get('config_count')}",
        f"signal_count={data.get('signal_count')}",
        "",
    ]
    for cfg in data.get("configs", []):
        lines.append(
            "Config #{idx} | {sym} {side} | date={date} close={close:.2f} | "
            "roc={roc:.4%} | downside_vol={dvol:.4%} | upside_vol={uvol:.4%} | "
            "pricing_vol_latest={pvol_latest:.4%} | pricing_vol_used={pvol_used:.4%}".format(
                idx=cfg.get("config_index"),
                sym=cfg.get("symbol"),
                side=str(cfg.get("configured_side", "")).upper(),
                date=cfg.get("date"),
                close=float(cfg.get("close", 0.0)),
                roc=float(cfg.get("latest_roc", cfg.get("roc", 0.0))),
                dvol=float(cfg.get("latest_downside_vol_annualized", cfg.get("downside_vol_annualized", 0.0))),
                uvol=float(cfg.get("latest_upside_vol_annualized", cfg.get("upside_vol_annualized", 0.0))),
                pvol_latest=float(cfg.get("latest_pricing_vol_annualized", cfg.get("pricing_vol_annualized", 0.0))),
                pvol_used=float(cfg.get("pricing_vol_annualized", 0.0)),
            )
        )
        for sig in cfg.get("fired_signals", []):
            lines.append(
                "SIGNAL: ENTER {side} POSITION | {sym} | expiry(next Friday in {dte} days) | "
                "estimated ATM premium={pps:.4f} per share ({ppc:.2f} per contract)".format(
                    side=str(sig.get("side", "")).upper(),
                    sym=cfg.get("symbol"),
                    dte=int(sig.get("days_to_expiry", 0)),
                    pps=float(sig.get("estimated_atm_premium_per_share", 0.0)),
                    ppc=float(sig.get("estimated_atm_premium_per_contract", 0.0)),
                )
            )
        lines.append("")

    excerpt = "\n".join(lines).rstrip()
    with open(args.github_output, "a", encoding="utf-8") as fh:
        fh.write(f"has_signal={'true' if has_signal else 'false'}\n")
        fh.write("excerpt<<EOF\n")
        fh.write(excerpt + "\n")
        fh.write("EOF\n")


if __name__ == "__main__":
    main()
