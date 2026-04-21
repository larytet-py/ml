import argparse
import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass
class Trade:
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_close: float
    exit_close: float
    strike: float
    premium: float
    intrinsic_at_expiry: float
    pnl_per_share: float
    pnl_per_contract: float
    roc_signal: float
    trend_vol_signal: float
    pricing_vol: float


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_put_price(spot: float, strike: float, time_to_expiry_years: float, risk_free_rate: float, sigma: float) -> float:
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


def black_scholes_call_price(spot: float, strike: float, time_to_expiry_years: float, risk_free_rate: float, sigma: float) -> float:
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


def downside_std(x) -> float:
    negatives = x[x < 0]
    if len(negatives) < 2:
        return 0.0
    return float(negatives.std(ddof=0))


def upside_std(x) -> float:
    positives = x[x > 0]
    if len(positives) < 2:
        return 0.0
    return float(positives.std(ddof=0))


def load_symbol_data(csv_path: str, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        raise ValueError(f"No rows found for symbol '{symbol}' in {csv_path}")

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "dividend_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after date filters.")
    return df


def run_backtest(
    df: pd.DataFrame,
    side: str,
    roc_lookback: int,
    put_roc_threshold: float,
    call_roc_threshold: float,
    vol_window: int,
    downside_vol_threshold_annualized: float,
    upside_vol_threshold_annualized: float,
    holding_days: int,
    risk_free_rate: float,
    min_pricing_vol_annualized: float,
    contract_size: int,
    allow_overlap: bool,
) -> pd.DataFrame:
    close = df["close"]
    returns = close.pct_change()

    df["roc"] = close / close.shift(roc_lookback) - 1.0
    df["downside_vol_annualized"] = (
        returns.rolling(vol_window).apply(downside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    df["upside_vol_annualized"] = (
        returns.rolling(vol_window).apply(upside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    df["pricing_vol_annualized"] = returns.rolling(vol_window).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)

    trades = []
    next_entry_idx = 0
    time_to_expiry = holding_days / TRADING_DAYS_PER_YEAR
    last_possible_entry = len(df) - holding_days - 1

    for i in range(len(df)):
        if i > last_possible_entry:
            break
        if i < next_entry_idx:
            continue

        row = df.iloc[i]
        if pd.isna(row["roc"]) or pd.isna(row["pricing_vol_annualized"]):
            continue

        if side == "put":
            if pd.isna(row["downside_vol_annualized"]):
                continue
            trigger = (
                row["roc"] <= put_roc_threshold
                and row["downside_vol_annualized"] >= downside_vol_threshold_annualized
            )
            trend_vol_signal = float(row["downside_vol_annualized"])
        else:
            if pd.isna(row["upside_vol_annualized"]):
                continue
            trigger = (
                row["roc"] >= call_roc_threshold
                and row["upside_vol_annualized"] >= upside_vol_threshold_annualized
            )
            trend_vol_signal = float(row["upside_vol_annualized"])

        if not trigger:
            continue

        entry_close = float(row["close"])
        strike = entry_close
        pricing_vol = max(float(row["pricing_vol_annualized"]), min_pricing_vol_annualized)
        if side == "put":
            premium = black_scholes_put_price(
                spot=entry_close,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                sigma=pricing_vol,
            )
        else:
            premium = black_scholes_call_price(
                spot=entry_close,
                strike=strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                sigma=pricing_vol,
            )

        exit_idx = i + holding_days
        exit_row = df.iloc[exit_idx]
        exit_close = float(exit_row["close"])
        intrinsic = max(strike - exit_close, 0.0) if side == "put" else max(exit_close - strike, 0.0)
        pnl_per_share = premium - intrinsic

        trades.append(
            Trade(
                side=side,
                entry_date=row["date"],
                exit_date=exit_row["date"],
                entry_close=entry_close,
                exit_close=exit_close,
                strike=strike,
                premium=premium,
                intrinsic_at_expiry=intrinsic,
                pnl_per_share=pnl_per_share,
                pnl_per_contract=pnl_per_share * contract_size,
                roc_signal=float(row["roc"]),
                trend_vol_signal=trend_vol_signal,
                pricing_vol=pricing_vol,
            )
        )

        if not allow_overlap:
            next_entry_idx = exit_idx + 1

    return pd.DataFrame([t.__dict__ for t in trades])


def print_summary(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        print("No trades generated with current settings.")
        return

    total = len(trades_df)
    wins = int((trades_df["pnl_per_share"] > 0).sum())
    total_pnl = trades_df["pnl_per_contract"].sum()
    avg_pnl = trades_df["pnl_per_contract"].mean()
    median_pnl = trades_df["pnl_per_contract"].median()
    avg_return_on_spot = (trades_df["pnl_per_share"] / trades_df["entry_close"]).mean()

    equity = trades_df["pnl_per_contract"].cumsum()
    running_peak = equity.cummax()
    max_drawdown = (equity - running_peak).min()

    print(f"Trades: {total}")
    print(f"Win rate: {wins / total:.2%}")
    print(f"Total PnL (per 1 contract): {total_pnl:.2f}")
    print(f"Average PnL/trade (per 1 contract): {avg_pnl:.2f}")
    print(f"Median PnL/trade (per 1 contract): {median_pnl:.2f}")
    print(f"Average return on spot notional: {avg_return_on_spot:.4%}")
    print(f"Max drawdown (per 1 contract, cumulative): {max_drawdown:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest: sell ATM weekly puts/calls after trend shock (ROC + directional stddev trigger)."
    )
    parser.add_argument("--csv", default="data/etfs.csv", help="Path to input CSV.")
    parser.add_argument("--symbol", required=True, help="Ticker to backtest, e.g. SPY.")
    parser.add_argument(
        "--side",
        choices=["put", "call"],
        default="put",
        help="`put`: sell after downtrend. `call`: sell after uptrend.",
    )
    parser.add_argument("--start-date", default=None, help="Optional filter, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional filter, YYYY-MM-DD.")
    parser.add_argument("--roc-lookback", type=int, default=5, help="Days for ROC.")
    parser.add_argument("--put-roc-threshold", type=float, default=-0.03, help="Put trigger: ROC <= threshold.")
    parser.add_argument("--call-roc-threshold", type=float, default=0.03, help="Call trigger: ROC >= threshold.")
    parser.add_argument("--vol-window", type=int, default=20, help="Rolling window for downside stddev.")
    parser.add_argument(
        "--downside-vol-threshold",
        type=float,
        default=0.20,
        help="Annualized downside vol threshold for put signal.",
    )
    parser.add_argument(
        "--upside-vol-threshold",
        type=float,
        default=0.20,
        help="Annualized upside vol threshold for call signal.",
    )
    parser.add_argument("--holding-days", type=int, default=5, help="Hold until this many trading days later.")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk-free rate for Black-Scholes.")
    parser.add_argument(
        "--min-pricing-vol",
        type=float,
        default=0.10,
        help="Vol floor (annualized) used when pricing the put premium.",
    )
    parser.add_argument("--contract-size", type=int, default=100, help="Shares per option contract.")
    parser.add_argument("--allow-overlap", action="store_true", help="Allow overlapping weekly positions.")
    parser.add_argument("--trades-out", default=None, help="Optional CSV path for trade log.")
    parser.add_argument("--print-trades", type=int, default=10, help="How many recent trades to print.")
    args = parser.parse_args()

    df = load_symbol_data(args.csv, args.symbol, args.start_date, args.end_date)
    trades_df = run_backtest(
        df=df,
        side=args.side,
        roc_lookback=args.roc_lookback,
        put_roc_threshold=args.put_roc_threshold,
        call_roc_threshold=args.call_roc_threshold,
        vol_window=args.vol_window,
        downside_vol_threshold_annualized=args.downside_vol_threshold,
        upside_vol_threshold_annualized=args.upside_vol_threshold,
        holding_days=args.holding_days,
        risk_free_rate=args.risk_free_rate,
        min_pricing_vol_annualized=args.min_pricing_vol,
        contract_size=args.contract_size,
        allow_overlap=args.allow_overlap,
    )

    print_summary(trades_df)

    if not trades_df.empty and args.print_trades > 0:
        print("\nRecent trades:")
        cols = [
            "side",
            "entry_date",
            "exit_date",
            "entry_close",
            "exit_close",
            "premium",
            "intrinsic_at_expiry",
            "pnl_per_contract",
            "roc_signal",
            "trend_vol_signal",
        ]
        print(trades_df[cols].tail(args.print_trades).to_string(index=False, justify="center"))

    if args.trades_out:
        trades_df.to_csv(args.trades_out, index=False)
        print(f"\nSaved trades to: {args.trades_out}")


if __name__ == "__main__":
    main()
