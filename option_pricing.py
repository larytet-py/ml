#!/usr/bin/env python3
import argparse
import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import Literal


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_put_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    sigma: float,
) -> float:
    if time_to_expiry_years <= 0:
        return max(strike - spot, 0.0)
    if sigma <= 1e-8:
        return max(strike * math.exp(-risk_free_rate * time_to_expiry_years) - spot, 0.0)

    sqrt_t = math.sqrt(time_to_expiry_years)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * sigma * sigma) * time_to_expiry_years
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return strike * math.exp(-risk_free_rate * time_to_expiry_years) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def black_scholes_call_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    sigma: float,
) -> float:
    if time_to_expiry_years <= 0:
        return max(spot - strike, 0.0)
    if sigma <= 1e-8:
        return max(spot - strike * math.exp(-risk_free_rate * time_to_expiry_years), 0.0)

    sqrt_t = math.sqrt(time_to_expiry_years)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * sigma * sigma) * time_to_expiry_years
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return spot * norm_cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry_years) * norm_cdf(d2)


class BlackScholesPricer:
    def __init__(self, risk_free_rate: float, min_sigma: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.min_sigma = min_sigma

    def effective_sigma(self, sigma: float) -> float:
        return max(float(sigma), self.min_sigma)

    def price(
        self,
        side: Literal["put", "call"],
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        sigma: float,
    ) -> float:
        used_sigma = self.effective_sigma(sigma)
        if side == "put":
            return black_scholes_put_price(
                spot=spot,
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=self.risk_free_rate,
                sigma=used_sigma,
            )
        if side == "call":
            return black_scholes_call_price(
                spot=spot,
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=self.risk_free_rate,
                sigma=used_sigma,
            )
        raise ValueError(f"Unsupported side '{side}'. Expected 'put' or 'call'.")

    def intrinsic_value(self, side: Literal["put", "call"], strike: float, spot: float) -> float:
        if side == "put":
            return max(strike - spot, 0.0)
        if side == "call":
            return max(spot - strike, 0.0)
        raise ValueError(f"Unsupported side '{side}'. Expected 'put' or 'call'.")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _annualized_sigma_from_closes(closes: list[float]) -> float | None:
    if len(closes) < 2:
        return None
    log_returns: list[float] = []
    prev = closes[0]
    for curr in closes[1:]:
        if prev <= 0.0 or curr <= 0.0:
            return None
        log_returns.append(math.log(curr / prev))
        prev = curr
    if len(log_returns) < 2:
        return None
    mean_r = sum(log_returns) / len(log_returns)
    var = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def _load_sigma_map(etf_csv_path: Path, vol_window: int) -> dict[tuple[str, date], float]:
    by_symbol: dict[str, list[tuple[date, float]]] = {}
    with etf_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            d = row.get("date")
            c = row.get("close")
            if not symbol or not d or not c:
                continue
            try:
                dt = _parse_date(d)
                close_val = float(c)
            except ValueError:
                continue
            by_symbol.setdefault(symbol, []).append((dt, close_val))

    sigma_map: dict[tuple[str, date], float] = {}
    for symbol, points in by_symbol.items():
        points.sort(key=lambda x: x[0])
        closes: list[float] = []
        for dt, close_val in points:
            closes.append(close_val)
            if len(closes) > vol_window:
                closes.pop(0)
            sigma = _annualized_sigma_from_closes(closes)
            if sigma is not None:
                sigma_map[(symbol, dt)] = sigma
    return sigma_map


def _format_money(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BS-estimated option OHLC values vs real option OHLC quotes."
    )
    parser.add_argument(
        "--options-csv",
        default="data/atm_options_2026_04_20-2026_04_27.csv",
        help="CSV with option and underlying OHLC quotes.",
    )
    parser.add_argument(
        "--etf-csv",
        default="data/etfs.csv",
        help="CSV with underlying ETF daily quotes used to estimate realized volatility.",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--vol-window", type=int, default=21)
    parser.add_argument(
        "--min-time-years",
        type=float,
        default=1.0 / 252.0,
        help="Minimum time to expiry used by BS model.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit printed rows (0 means all rows).")
    args = parser.parse_args()

    sigma_map = _load_sigma_map(Path(args.etf_csv), vol_window=args.vol_window)
    pricer = BlackScholesPricer(risk_free_rate=args.risk_free_rate)

    columns = [
        ("symbol", 6),
        ("date", 10),
        ("side", 4),
        ("strike", 8),
        ("sigma", 7),
        ("open_m", 8),
        ("open_r", 8),
        ("close_m", 8),
        ("close_r", 8),
        ("high_m", 8),
        ("high_r", 8),
        ("low_m", 8),
        ("low_r", 8),
    ]
    header = " ".join(name.rjust(width) for name, width in columns)
    print(header)
    print("-" * len(header))

    printed = 0
    skipped = 0
    with Path(args.options_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.limit > 0 and printed >= args.limit:
                break
            if row.get("missing_reason"):
                skipped += 1
                continue
            symbol = (row.get("symbol") or "").strip()
            d_raw = row.get("date")
            exp_raw = row.get("expiration_date")
            side = (row.get("contract_type") or "").strip().lower()
            strike_raw = row.get("strike_price")
            if not symbol or not d_raw or not exp_raw or side not in ("call", "put") or not strike_raw:
                skipped += 1
                continue
            try:
                trade_date = _parse_date(d_raw)
                expiration_date = _parse_date(exp_raw)
                strike = float(strike_raw)
                sigma = sigma_map[(symbol, trade_date)]
                underlying_open = float(row["underlying_open"])
                underlying_close = float(row["underlying_close"])
                underlying_high = float(row["underlying_high"])
                underlying_low = float(row["underlying_low"])
                option_open = float(row["option_open"])
                option_close = float(row["option_close"])
                option_high = float(row["option_high"])
                option_low = float(row["option_low"])
            except (KeyError, ValueError):
                skipped += 1
                continue
            except Exception:
                skipped += 1
                continue

            dte_days = max((expiration_date - trade_date).days, 0)
            t_years = max(dte_days / 365.0, float(args.min_time_years))
            model_open = pricer.price(side, underlying_open, strike, t_years, sigma)
            model_close = pricer.price(side, underlying_close, strike, t_years, sigma)
            model_high = pricer.price(side, underlying_high, strike, t_years, sigma)
            model_low = pricer.price(side, underlying_low, strike, t_years, sigma)

            values = [
                symbol,
                d_raw,
                side,
                f"{strike:.2f}",
                f"{sigma:.3f}",
                _format_money(model_open),
                _format_money(option_open),
                _format_money(model_close),
                _format_money(option_close),
                _format_money(model_high),
                _format_money(option_high),
                _format_money(model_low),
                _format_money(option_low),
            ]
            row_text = " ".join(v.rjust(width) for v, (_, width) in zip(values, columns))
            print(row_text)
            printed += 1

    print()
    print(f"Printed rows: {printed}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
