#!/usr/bin/env python3
"""
Usage:

MARKET_STACK_API_KEY=1fd.. python3 option_signal_notifier.py

MARKET_STACK_API_KEY=1fd.. python3 option_signal_notifier.py \
  --config data/option_signal_notifier.yml
"""
import argparse
import json
import os
import shlex
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from option_signal_config import load_signal_strategy_dicts
from option_pricing import black_scholes_call_price, black_scholes_put_price
from weekly_option_reversal_core import build_signal_frame
from weekly_option_acceleration_core import build_acceleration_signal_frame
from weekly_option_roc_accel_core import build_roc_accel_signal_frame


DEFAULT_CACHE_CSV = "data/etfs.csv"
CANONICAL_DATA_COLUMNS = ["symbol", "date", "open", "close", "high", "low", "volume", "dividend_rate"]
ATM_OPTIONS_FIELDNAMES = [
    "symbol",
    "date",
    "underlying_open",
    "underlying_high",
    "underlying_low",
    "underlying_close",
    "underlying_volume",
    "contract_type",
    "option_ticker",
    "expiration_date",
    "strike_price",
    "option_open",
    "option_high",
    "option_low",
    "option_close",
    "option_volume",
    "option_vwap",
    "option_transactions",
    "missing_reason",
]
CONSERVATIVE_LIMIT_PREMIUM_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "GDX": {"call": 0.30, "put": 0.75},
    "IBIT": {"call": 0.20, "put": 0.10},
    "IWM": {"call": 0.20, "put": 0.75},
    "QQQ": {"call": -0.20, "put": 0.50},
    "SPY": {"call": -0.30, "put": 0.30},
    "TLT": {"call": 0.05, "put": 0.15},
    "VXX": {"call": 0.20, "put": -0.05},
}
CONSERVATIVE_LIMIT_TARGET_FILL_PROBABILITY = 0.50


def _unix_to_yyyy_mm_dd(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    try:
        value = int(float(ts))
    except (TypeError, ValueError):
        return None
    return datetime.utcfromtimestamp(value).date().isoformat()


def _marketdata_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    token = os.getenv("MARKET_DATA_TOKEN")
    if not token:
        raise RuntimeError("Missing MARKET_DATA_TOKEN environment variable.")

    url = f"https://api.marketdata.app{path}"
    req_params = params or {}

    def _request_to_curl(req_url: str, query_params: Dict[str, Any]) -> str:
        curl_parts = ["curl", "-sS", shlex.quote(req_url), "-H", shlex.quote("Authorization: Bearer $MARKET_DATA_TOKEN")]
        curl_parts.extend(["-H", shlex.quote("Accept: application/json")])
        for key, value in query_params.items():
            curl_parts.extend(["--data-urlencode", shlex.quote(f"{key}={value}")])
        return " ".join(curl_parts)

    print(f"[marketdata] request: {_request_to_curl(url, req_params)}")
    response = requests.get(
        url,
        params=req_params,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        body_preview = response.text[:240].replace("\n", " ")
        print(
            "[marketdata] response: "
            f"status={response.status_code} error_body={json.dumps(body_preview)}"
        )
        raise
    payload = response.json()
    if isinstance(payload, dict):
        row_count = len(payload.get("t") or [])
        if row_count == 0:
            for list_key in ("optionSymbol", "strike", "expiration", "side"):
                values = payload.get(list_key)
                if isinstance(values, list):
                    row_count = len(values)
                    break
        summary = {
            "s": payload.get("s"),
            "keys": sorted(payload.keys()),
            "rows": row_count,
        }
        print(
            "[marketdata] response: "
            f"status={response.status_code} "
            f"summary={json.dumps(summary, sort_keys=True)}"
        )
    else:
        print(f"[marketdata] response: status={response.status_code} payload_type={type(payload).__name__}")

    if isinstance(payload, dict) and payload.get("s") == "error":
        raise RuntimeError(f"MarketData.app API error for {path}: {payload.get('errmsg', 'unknown error')}")
    return payload


def _marketdata_candles_rows(symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    payload = _marketdata_get(
        f"/v1/stocks/candles/D/{symbol.upper()}/",
        params={"from": start_date, "to": end_date},
    )
    if payload.get("s") == "no_data":
        return []

    times = payload.get("t") or []
    opens = payload.get("o") or []
    highs = payload.get("h") or []
    lows = payload.get("l") or []
    closes = payload.get("c") or []
    volumes = payload.get("v") or []

    rows: List[Dict[str, Any]] = []
    for i, ts in enumerate(times):
        rows.append(
            {
                "symbol": symbol.upper(),
                "date": _unix_to_yyyy_mm_dd(ts),
                "underlying_open": opens[i] if i < len(opens) else None,
                "underlying_high": highs[i] if i < len(highs) else None,
                "underlying_low": lows[i] if i < len(lows) else None,
                "underlying_close": closes[i] if i < len(closes) else None,
                "underlying_volume": volumes[i] if i < len(volumes) else None,
            }
        )
    return [r for r in rows if r["date"]]


def _choose_atm_from_chain(
    chain_payload: Dict[str, Any],
    side: str,
    spot: float,
    trade_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    option_symbols = chain_payload.get("optionSymbol") or []
    sides = chain_payload.get("side") or []
    strikes = chain_payload.get("strike") or []
    expirations = chain_payload.get("expiration") or []

    trade_date_obj = None
    if trade_date:
        try:
            trade_date_obj = datetime.strptime(trade_date, "%Y-%m-%d").date()
        except ValueError:
            trade_date_obj = None

    candidates: List[Dict[str, Any]] = []
    for i, option_symbol in enumerate(option_symbols):
        option_side = str(sides[i]).lower() if i < len(sides) else ""
        if option_side != side:
            continue
        strike = strikes[i] if i < len(strikes) else None
        expiration = expirations[i] if i < len(expirations) else None
        if strike is None:
            continue
        try:
            strike_float = float(strike)
        except (TypeError, ValueError):
            continue

        expiration_date = _unix_to_yyyy_mm_dd(expiration)
        if trade_date_obj and expiration_date:
            try:
                expiration_date_obj = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            except ValueError:
                expiration_date_obj = None
            # Prefer unexpired contracts for snapshot/candle requests made after market close.
            if expiration_date_obj is not None and expiration_date_obj <= trade_date_obj:
                continue

        candidates.append(
            {
                "optionSymbol": option_symbol,
                "strike": strike_float,
                "expiration": expiration,
                "expiration_date": expiration_date,
            }
        )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda c: (
            c["expiration"] if c["expiration"] is not None else 10**18,
            abs(c["strike"] - spot),
            c["strike"],
            c["optionSymbol"],
        ),
    )


def _empty_atm_option_row(underlying_row: Dict[str, Any], side: str, reason: str) -> Dict[str, Any]:
    return {
        **underlying_row,
        "contract_type": side,
        "option_ticker": None,
        "expiration_date": None,
        "strike_price": None,
        "option_open": None,
        "option_high": None,
        "option_low": None,
        "option_close": None,
        "option_volume": None,
        "option_vwap": None,
        "option_transactions": None,
        "missing_reason": reason,
    }


def _nearest_strike_reference(price: float) -> float:
    # Use half-up rounding so .5 always rounds away from zero.
    return float(int(price + 0.5)) if price >= 0 else float(int(price - 0.5))


def _previous_close_on_or_before(cache_df: pd.DataFrame, trade_date: str) -> Optional[float]:
    if cache_df.empty:
        return None
    trade_ts = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(trade_ts):
        return None
    eligible = cache_df[cache_df["date"] < trade_ts]
    if eligible.empty:
        return None
    value = eligible.iloc[-1].get("close")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _fetch_marketdata_option_candle(option_symbol: str, trade_date: str) -> Optional[Dict[str, Any]]:
    try:
        payload = _marketdata_get(
            f"/v1/options/candles/D/{option_symbol}/",
            params={"from": trade_date, "to": trade_date},
        )
    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 404:
            return None
        raise
    if payload.get("s") == "no_data":
        return None

    times = payload.get("t") or []
    if not times:
        return None
    return {
        "option_open": (payload.get("o") or [None])[0],
        "option_high": (payload.get("h") or [None])[0],
        "option_low": (payload.get("l") or [None])[0],
        "option_close": (payload.get("c") or [None])[0],
        "option_volume": (payload.get("v") or [None])[0],
    }


def _fetch_marketdata_option_quote_snapshot(option_symbol: str, trade_date: str) -> Dict[str, Any]:
    try:
        payload = _marketdata_get(f"/v1/options/quotes/{option_symbol}/", params={"date": trade_date})
    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        body = ""
        if response is not None:
            try:
                body = (response.text or "").lower()
            except Exception:
                body = ""
        if response is not None and response.status_code == 404:
            return {}
        # Some entitlements reject historical quote requests using `date`.
        if response is not None and response.status_code == 400 and "date parameter is used for historical queries" in body:
            payload = _marketdata_get(f"/v1/options/quotes/{option_symbol}/")
        else:
            raise
    if payload.get("s") == "no_data":
        return {}

    return {
        "option_vwap": (payload.get("mid") or [None])[0],
        "option_transactions": None,
    }


def _fetch_marketdata_option_chain(symbol: str, trade_date: str, expiration_to: str) -> Dict[str, Any]:
    # MarketData can reject `date` for non-historical accounts; fetch a filtered current chain instead.
    return _marketdata_get(
        f"/v1/options/chain/{symbol}/",
        params={
            "from": trade_date,
            "to": expiration_to,
            "weekly": "true",
            "monthly": "true",
            "quarterly": "true",
        },
    )


def _update_atm_options_csv_from_marketdata(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    out_csv_path: str = "data/atm_options.csv",
    expiry_lookahead_days: int = 14,
) -> Dict[str, Any]:
    """
    Fetch ATM call/put daily option data from MarketData.app and overwrite atm options CSV.

    Auth: reads bearer token from MARKET_DATA_TOKEN env var.
    """
    normalized_symbols = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    if not normalized_symbols:
        raise ValueError("symbols cannot be empty")

    output_rows: List[Dict[str, Any]] = []

    for symbol in normalized_symbols:
        symbol_cache_df = pd.DataFrame(columns=CANONICAL_DATA_COLUMNS)
        cache_path = Path(DEFAULT_CACHE_CSV)
        if cache_path.exists():
            try:
                symbol_cache_df = load_symbol_data_from_csv(DEFAULT_CACHE_CSV, symbol, None, None)
                symbol_cache_df = symbol_cache_df.sort_values("date").reset_index(drop=True)
            except Exception:
                symbol_cache_df = pd.DataFrame(columns=CANONICAL_DATA_COLUMNS)

        try:
            underlying_rows = _marketdata_candles_rows(symbol, start_date, end_date)
        except Exception as exc:
            print(f"Warning: skipping {symbol} ATM refresh due to underlying candles error: {exc}")
            continue
        for underlying_row in underlying_rows:
            spot = underlying_row.get("underlying_close")
            trade_date = underlying_row.get("date")
            if trade_date is None:
                continue
            if spot is None:
                output_rows.append(_empty_atm_option_row(underlying_row, "call", "missing_underlying_close"))
                output_rows.append(_empty_atm_option_row(underlying_row, "put", "missing_underlying_close"))
                continue

            trade_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
            expiration_to = (trade_dt + timedelta(days=expiry_lookahead_days)).isoformat()
            try:
                chain = _fetch_marketdata_option_chain(symbol, trade_date=trade_date, expiration_to=expiration_to)
            except Exception:
                output_rows.append(_empty_atm_option_row(underlying_row, "call", "chain_request_failed"))
                output_rows.append(_empty_atm_option_row(underlying_row, "put", "chain_request_failed"))
                continue
            reference_spot = _previous_close_on_or_before(symbol_cache_df, trade_date)
            if reference_spot is None:
                reference_spot = float(spot)
            rounded_reference_spot = _nearest_strike_reference(reference_spot)

            for side in ("call", "put"):
                contract = _choose_atm_from_chain(
                    chain,
                    side=side,
                    spot=rounded_reference_spot,
                    trade_date=trade_date,
                )
                if not contract:
                    output_rows.append(_empty_atm_option_row(underlying_row, side, "no_contract_found"))
                    continue

                option_symbol = str(contract["optionSymbol"])
                candle = _fetch_marketdata_option_candle(option_symbol, trade_date)
                quote = _fetch_marketdata_option_quote_snapshot(option_symbol, trade_date)

                if candle is None:
                    output_rows.append(
                        {
                            **underlying_row,
                            "contract_type": side,
                            "option_ticker": option_symbol,
                            "expiration_date": contract.get("expiration_date"),
                            "strike_price": contract.get("strike"),
                            "option_open": None,
                            "option_high": None,
                            "option_low": None,
                            "option_close": None,
                            "option_volume": None,
                            "option_vwap": quote.get("option_vwap"),
                            "option_transactions": quote.get("option_transactions"),
                            "missing_reason": "no_option_bar",
                        }
                    )
                    continue

                output_rows.append(
                    {
                        **underlying_row,
                        "contract_type": side,
                        "option_ticker": option_symbol,
                        "expiration_date": contract.get("expiration_date"),
                        "strike_price": contract.get("strike"),
                        "option_open": candle.get("option_open"),
                        "option_high": candle.get("option_high"),
                        "option_low": candle.get("option_low"),
                        "option_close": candle.get("option_close"),
                        "option_volume": candle.get("option_volume"),
                        "option_vwap": quote.get("option_vwap"),
                        "option_transactions": quote.get("option_transactions"),
                        "missing_reason": None,
                    }
                )

    output_df = pd.DataFrame(output_rows)
    if output_df.empty:
        raise RuntimeError("No rows returned from MarketData.app for the requested symbols/date range.")

    output_df = output_df[ATM_OPTIONS_FIELDNAMES].sort_values(["symbol", "date", "contract_type"]).reset_index(drop=True)
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            for col in ATM_OPTIONS_FIELDNAMES:
                if col not in existing_df.columns:
                    existing_df[col] = pd.NA
            existing_df = existing_df[ATM_OPTIONS_FIELDNAMES]
        except Exception:
            existing_df = pd.DataFrame(columns=ATM_OPTIONS_FIELDNAMES)
    else:
        existing_df = pd.DataFrame(columns=ATM_OPTIONS_FIELDNAMES)

    existing_by_key = existing_df.set_index(["symbol", "date", "contract_type"], drop=False)
    output_by_key = output_df.set_index(["symbol", "date", "contract_type"], drop=False)
    existing_by_key.update(output_by_key)
    new_rows = output_by_key.loc[~output_by_key.index.isin(existing_by_key.index)]
    merged_df = pd.concat([existing_by_key, new_rows], ignore_index=True)
    merged_df = merged_df.sort_values(["symbol", "date", "contract_type"]).reset_index(drop=True)
    merged_df.to_csv(out_path, index=False)
    return {
        "merged_df": merged_df,
        "fetched_rows": len(output_df),
        "total_rows": len(merged_df),
    }


def _recommended_conservative_limit_premium(symbol: str, side: str, modeled_premium: float) -> Optional[float]:
    symbol_adjustments = CONSERVATIVE_LIMIT_PREMIUM_ADJUSTMENTS.get(symbol.upper())
    if not symbol_adjustments:
        return None
    side_adjustment = symbol_adjustments.get(side.lower())
    if side_adjustment is None:
        return None
    return modeled_premium + side_adjustment


@dataclass
class SignalConfig:
    name: str
    note: Optional[str]
    symbol: str
    side: str
    signal_model: str
    roc_lookback: int
    accel_window: int
    vol_window: int
    roc_comparator: Optional[str]
    roc_threshold: Optional[float]
    accel_comparator: Optional[str]
    accel_threshold: Optional[float]
    vol_comparator: Optional[str]
    vol_threshold: Optional[float]


def _compare(value: Optional[float], comparator: Optional[str], threshold: Optional[float]) -> bool:
    if comparator is None or threshold is None:
        return True
    if value is None:
        return False
    if comparator == "above":
        return value >= threshold
    return value <= threshold


def _normalize_market_data_columns(df: pd.DataFrame, symbol_hint: Optional[str] = None) -> pd.DataFrame:
    normalized = df.copy()
    if "date" not in normalized.columns and "Date" in normalized.columns:
        normalized = normalized.rename(columns={"Date": "date"})
    for old_name, new_name in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
        if old_name in normalized.columns and new_name not in normalized.columns:
            normalized = normalized.rename(columns={old_name: new_name})

    if "symbol" not in normalized.columns:
        if symbol_hint is None:
            raise ValueError("Input data must contain a 'symbol' column when no symbol hint is provided.")
        normalized["symbol"] = symbol_hint.upper()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()

    if "date" not in normalized.columns:
        raise ValueError("Input data must contain a 'date' column.")
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.tz_localize(None)

    for col in ["open", "close", "high", "low", "volume", "dividend_rate"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        else:
            normalized[col] = pd.NA

    return normalized[CANONICAL_DATA_COLUMNS]


def load_symbol_data_from_csv(csv_path: str, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    df = _normalize_market_data_columns(pd.read_csv(csv_path), symbol_hint=symbol)
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        raise ValueError(f"No rows found for symbol '{symbol}' in {csv_path}")

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    # Filter out invalid placeholder rows (for example close=0) that break return/volatility math.
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows left after date/price filters.")
    return df


def _fetch_marketstack_daily(symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    api_key = os.getenv("MARKET_STACK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MARKET_STACK_API_KEY environment variable.")

    base_url = "http://api.marketstack.com/v2/eod"
    limit = 1000
    offset = 0
    rows = []

    def _request_to_curl(url: str, req_params: Dict[str, Any]) -> str:
        curl_parts = ["curl", "-sS", shlex.quote(url)]
        for key, value in req_params.items():
            if key == "access_key":
                data_value = f"{key}=$MARKET_STACK_API_KEY"
            else:
                data_value = f"{key}={value}"
            curl_parts.extend(["--data-urlencode", shlex.quote(data_value)])
        return " ".join(curl_parts)

    while True:
        params = {
            "access_key": api_key,
            "symbols": symbol.upper(),
            "limit": limit,
            "offset": offset,
            "sort": "ASC",
        }
        if start_date:
            params["date_from"] = start_date
        if end_date:
            params["date_to"] = end_date

        print(f"[marketstack] request: {_request_to_curl(base_url, params)}")
        try:
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download daily data for {symbol} from marketstack: {exc}") from exc

        payload = response.json()
        response_batch = payload.get("data", [])
        response_pagination = payload.get("pagination", {})
        print(
            "[marketstack] response: "
            f"status={response.status_code} "
            f"rows={len(response_batch)} "
            f"pagination={json.dumps(response_pagination, sort_keys=True)}"
        )
        if "error" in payload:
            message = payload["error"].get("message", "Unknown marketstack error")
            raise RuntimeError(f"marketstack API error for {symbol}: {message}")

        batch = response_batch
        if not batch:
            break
        rows.extend(batch)

        pagination = response_pagination
        count = int(pagination.get("count", len(batch)))
        total = int(pagination.get("total", len(rows)))
        if count == 0 or offset + count >= total:
            break
        offset += count

    if not rows:
        raise RuntimeError(f"No data returned for symbol {symbol} from marketstack.")

    df = _normalize_market_data_columns(pd.DataFrame(rows), symbol_hint=symbol.upper())
    if df["date"].isna().all():
        raise RuntimeError(f"marketstack response for {symbol} did not include parseable 'date' values.")
    if df["close"].isna().all():
        raise RuntimeError(f"marketstack response for {symbol} did not include 'close'.")

    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _is_recent_enough(latest_date: pd.Timestamp) -> bool:
    # Require cache to include the latest completed business day.
    # In practice this means only "today" can be missing.
    today = pd.Timestamp.today().normalize()
    latest_expected_trading_day = (today - pd.offsets.BDay(1)).normalize()
    return latest_date.normalize() >= latest_expected_trading_day


def _persist_symbol_rows(cache_csv_path: str, symbol: str, fresh_rows: pd.DataFrame) -> None:
    cache_path = Path(cache_csv_path)
    if cache_path.exists():
        full_df = _normalize_market_data_columns(pd.read_csv(cache_csv_path), symbol_hint=symbol)
    else:
        full_df = pd.DataFrame(columns=CANONICAL_DATA_COLUMNS)

    keep_mask = full_df["symbol"].astype(str).str.upper() != symbol.upper()
    other_symbols = full_df[keep_mask].copy()

    symbol_df = _normalize_market_data_columns(fresh_rows.copy(), symbol_hint=symbol)
    symbol_df = symbol_df.dropna(subset=["date", "close"])
    symbol_df = symbol_df[symbol_df["close"] > 0]
    symbol_df = symbol_df.drop_duplicates(subset=["date"], keep="last")
    symbol_df = symbol_df.sort_values("date")

    merged = pd.concat([other_symbols, symbol_df], ignore_index=True)
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    merged = merged[CANONICAL_DATA_COLUMNS]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_csv_path, index=False)


def _load_or_refresh_cached_data(
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    cache_csv_path: str,
) -> Optional[pd.DataFrame]:
    cache_path = Path(cache_csv_path)
    if not cache_path.exists():
        return None

    try:
        cached = load_symbol_data_from_csv(cache_csv_path, symbol, None, None)
    except ValueError:
        return None

    if cached.empty:
        return None

    latest_cached_date = pd.Timestamp(cached["date"].max()).tz_localize(None)
    if _is_recent_enough(latest_cached_date):
        print(f"Using cached data from {cache_csv_path} (latest {latest_cached_date.date()}); skipping marketstack API.")
        return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)

    api_key = os.getenv("MARKET_STACK_API_KEY")
    if not api_key:
        print(
            f"Cached data in {cache_csv_path} for {symbol.upper()} is stale (latest {latest_cached_date.date()}) and "
            "MARKET_STACK_API_KEY is not set; using cache without refresh."
        )
        return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)

    missing_from = (latest_cached_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    target_to = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        fresh = _fetch_marketstack_daily(symbol, start_date=missing_from, end_date=target_to)
    except RuntimeError as exc:
        if "No data returned for symbol" in str(exc):
            print(
                f"Warning: cache refresh returned no rows for {symbol.upper()} in range "
                f"{missing_from}..{target_to}; using existing cached data."
            )
        else:
            print(f"Cache refresh failed ({exc}); using existing cached data.")
        return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)

    if fresh.empty:
        print(
            f"Warning: cache refresh returned empty dataframe for {symbol.upper()} in range "
            f"{missing_from}..{target_to}; using existing cached data."
        )
        return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)

    combined = pd.concat([cached, fresh], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.tz_localize(None)
    combined = combined.dropna(subset=["date", "close"])
    combined = combined[combined["close"] > 0]
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    _persist_symbol_rows(cache_csv_path, symbol, combined)

    updated_last = pd.Timestamp(combined["date"].max()).date()
    print(f"Updated {cache_csv_path} for {symbol.upper()} through {updated_last}.")
    return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)


def load_symbol_data(symbol: str, start_date: Optional[str], end_date: Optional[str], csv_path: Optional[str]) -> pd.DataFrame:
    symbol = symbol.upper()
    if csv_path:
        return load_symbol_data_from_csv(csv_path, symbol, start_date, end_date)

    cached = _load_or_refresh_cached_data(symbol, start_date, end_date, DEFAULT_CACHE_CSV)
    if cached is not None and not cached.empty:
        return cached.reset_index(drop=True)

    df = _fetch_marketstack_daily(symbol, start_date, end_date)
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    if df.empty:
        raise ValueError("No rows left after date filters.")

    # Seed local cache for future runs when it is absent.
    try:
        _persist_symbol_rows(DEFAULT_CACHE_CSV, symbol, df)
        print(f"Saved {len(df)} rows to {DEFAULT_CACHE_CSV} for future runs.")
    except Exception as exc:  # pragma: no cover - non-fatal cache write path
        print(f"Warning: failed to persist cache to {DEFAULT_CACHE_CSV}: {exc}")

    return df.reset_index(drop=True)


def _load_configs(args: argparse.Namespace) -> List[SignalConfig]:
    def _validate_metric_tuples(cfg: SignalConfig) -> None:
        tuples = [
            ("roc", cfg.roc_comparator, cfg.roc_threshold),
            ("accel", cfg.accel_comparator, cfg.accel_threshold),
            ("vol", cfg.vol_comparator, cfg.vol_threshold),
        ]
        complete = 0
        for metric, comp, thr in tuples:
            if (comp is None) ^ (thr is None):
                raise ValueError(
                    f"{cfg.symbol} {cfg.side}: {metric} comparator/threshold must be provided together."
                )
            if comp is not None and thr is not None:
                complete += 1
        if complete == 0:
            raise ValueError(
                f"{cfg.symbol} {cfg.side}: specify at least one tuple like --roc-comparator above --roc-threshold 0.3."
            )
        if cfg.signal_model == "roc" and cfg.accel_comparator is not None:
            raise ValueError(f"{cfg.symbol} {cfg.side}: accel tuple is not valid for signal-model=roc.")
        if cfg.signal_model == "accel" and cfg.roc_comparator is not None:
            raise ValueError(f"{cfg.symbol} {cfg.side}: roc tuple is not valid for signal-model=accel.")

    configs: List[SignalConfig] = []
    for row in load_signal_strategy_dicts(args.config):
        cfg = SignalConfig(
            name=row["name"],
            note=row.get("note"),
            symbol=row["symbol"],
            side=row["side"],
            signal_model="roc",
            roc_lookback=row["roc_window_size"],
            accel_window=row["accel_window"],
            vol_window=row["vol_window_size"],
            roc_comparator=row["roc_comparator"],
            roc_threshold=row["roc_threshold"],
            accel_comparator=row["accel_comparator"],
            accel_threshold=row["accel_threshold"],
            vol_comparator=row["vol_comparator"],
            vol_threshold=row["vol_threshold"],
        )
        _validate_metric_tuples(cfg)
        configs.append(cfg)

    return configs


def _required_history(cfg: SignalConfig) -> int:
    if cfg.signal_model == "accel-roc":
        return max(cfg.roc_lookback + 1, cfg.accel_window + 2)
    if cfg.signal_model == "accel":
        return max(cfg.accel_window + 2, cfg.vol_window + 1)
    return max(cfg.roc_lookback + 1, cfg.vol_window + 1)


def _evaluate_config(
    cfg: SignalConfig,
    symbol_df: pd.DataFrame,
    risk_free_rate: float,
    min_pricing_vol: float,
    contract_size: int,
) -> Dict[str, Any]:
    if cfg.signal_model == "accel":
        signal_df = build_acceleration_signal_frame(symbol_df, cfg.accel_window, cfg.vol_window)
    elif cfg.signal_model == "roc":
        signal_df = build_signal_frame(symbol_df, cfg.roc_lookback, cfg.vol_window)
    else:
        signal_df = build_roc_accel_signal_frame(symbol_df, cfg.roc_lookback, cfg.accel_window)

    latest = signal_df.iloc[-1]
    latest_roc = float(latest["roc"]) if "roc" in signal_df.columns and not pd.isna(latest["roc"]) else None
    latest_acceleration = (
        float(latest["acceleration"])
        if "acceleration" in signal_df.columns and not pd.isna(latest["acceleration"])
        else None
    )
    latest_pricing_vol_value = latest["pricing_vol_annualized"]
    if (
        pd.isna(latest_pricing_vol_value)
        or (cfg.signal_model in {"roc", "accel-roc"} and latest_roc is None)
        or (cfg.signal_model in {"accel", "accel-roc"} and latest_acceleration is None)
    ):
        needed = _required_history(cfg)
        raise ValueError(
            f"Insufficient history for {cfg.symbol} {cfg.side} with model/window "
            f"({cfg.signal_model}, {cfg.vol_window}). Need at least {needed} rows, got {len(signal_df)} rows."
        )

    if cfg.side == "put":
        selected_sides: List[str] = ["put"]
    elif cfg.side == "call":
        selected_sides = ["call"]
    else:
        selected_sides = ["put", "call"]

    spot = float(latest["close"])
    latest_date = pd.Timestamp(latest["date"])
    latest_downside_vol = (
        float(latest["downside_vol_annualized"])
        if "downside_vol_annualized" in signal_df.columns and not pd.isna(latest["downside_vol_annualized"])
        else None
    )
    latest_upside_vol = (
        float(latest["upside_vol_annualized"])
        if "upside_vol_annualized" in signal_df.columns and not pd.isna(latest["upside_vol_annualized"])
        else None
    )
    latest_pricing_vol = float(latest_pricing_vol_value)
    pricing_vol = max(latest_pricing_vol, min_pricing_vol)

    weekday = latest_date.weekday()
    days_to_expiry = 4 - weekday
    has_time = days_to_expiry > 0
    put_trigger = False
    call_trigger = False
    if has_time and "put" in selected_sides:
        put_trigger = (
            _compare(latest_roc, cfg.roc_comparator, cfg.roc_threshold)
            and _compare(latest_acceleration, cfg.accel_comparator, cfg.accel_threshold)
            and _compare(latest_downside_vol, cfg.vol_comparator, cfg.vol_threshold)
        )
    if has_time and "call" in selected_sides:
        call_trigger = (
            _compare(latest_roc, cfg.roc_comparator, cfg.roc_threshold)
            and _compare(latest_acceleration, cfg.accel_comparator, cfg.accel_threshold)
            and _compare(latest_upside_vol, cfg.vol_comparator, cfg.vol_threshold)
        )

    messages: List[str] = []
    fired_signals: List[Dict[str, Any]] = []
    fired = False
    for side in selected_sides:
        is_triggered = put_trigger if side == "put" else call_trigger
        if not is_triggered:
            continue
        time_to_expiry_years = days_to_expiry / 365.25

        if side == "put":
            premium = black_scholes_put_price(
                spot=spot,
                strike=spot,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=risk_free_rate,
                sigma=pricing_vol,
            )
        else:
            premium = black_scholes_call_price(
                spot=spot,
                strike=spot,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=risk_free_rate,
                sigma=pricing_vol,
            )

        fired = True
        premium_per_contract = premium * contract_size
        recommended_limit = _recommended_conservative_limit_premium(cfg.symbol, side, premium)
        recommended_limit_text = (
            f", recommended conservative limit={recommended_limit:.4f} per share "
            f"(target fill probability={CONSERVATIVE_LIMIT_TARGET_FILL_PROBABILITY:.0%})"
            if recommended_limit is not None
            else ""
        )
        signal_message = (
            f"ENTER {side.upper()} POSITION | {cfg.symbol} | next Friday in {days_to_expiry} days | "
            f"estimated ATM premium={premium:.4f} per share{recommended_limit_text}, last close {spot:.2f}"
        )
        messages.append(signal_message)
        fired_signals.append(
            {
                "side": side,
                "action": f"enter_{side}",
                "message": signal_message,
                "days_to_expiry": days_to_expiry,
                "estimated_atm_premium_per_share": premium,
                "estimated_atm_premium_per_contract": premium_per_contract,
                "recommended_conservative_limit_per_share": recommended_limit,
                "recommended_aggressive_limit_per_share": recommended_limit,
                "target_fill_probability": (
                    CONSERVATIVE_LIMIT_TARGET_FILL_PROBABILITY if recommended_limit is not None else None
                ),
            }
        )

    if not fired:
        pass

    result: Dict[str, Any] = {
        "messages": messages,
        "strategy_name": cfg.name,
        "strategy_note": cfg.note,
        "symbol": cfg.symbol,
        "configured_side": cfg.side,
        "signal_model": cfg.signal_model,
        "selected_sides": selected_sides,
        "roc_lookback": cfg.roc_lookback,
        "accel_window": cfg.accel_window,
        "vol_window": cfg.vol_window,
        "date": latest_date.date().isoformat(),
        "close": spot,
        "roc": latest_roc,
        "acceleration": latest_acceleration,
        # Backward-compatible field: pricing vol actually used for premium estimation (after floor).
        "pricing_vol_annualized": pricing_vol,
        # Explicit raw/latest calculated values from the latest bar.
        "latest_roc": latest_roc,
        "latest_acceleration": latest_acceleration,
        "latest_pricing_vol_annualized": latest_pricing_vol,
        "fired_signals": fired_signals,
    }

    if "put" in selected_sides:
        put_fields: Dict[str, Any] = {
            "downside_vol_annualized": latest_downside_vol,
            "put_trigger": put_trigger,
            "vol_comparator": cfg.vol_comparator,
            "vol_threshold": cfg.vol_threshold,
        }
        if cfg.roc_comparator is not None:
            put_fields["roc_comparator"] = cfg.roc_comparator
            put_fields["roc_threshold"] = cfg.roc_threshold
        if cfg.accel_comparator is not None:
            put_fields["accel_comparator"] = cfg.accel_comparator
            put_fields["accel_threshold"] = cfg.accel_threshold
        result.update(put_fields)
    if "call" in selected_sides:
        call_fields: Dict[str, Any] = {
            "upside_vol_annualized": latest_upside_vol,
            "call_trigger": call_trigger,
            "vol_comparator": cfg.vol_comparator,
            "vol_threshold": cfg.vol_threshold,
        }
        if cfg.roc_comparator is not None:
            call_fields["roc_comparator"] = cfg.roc_comparator
            call_fields["roc_threshold"] = cfg.roc_threshold
        if cfg.accel_comparator is not None:
            call_fields["accel_comparator"] = cfg.accel_comparator
            call_fields["accel_threshold"] = cfg.accel_threshold
        result.update(call_fields)
    return result


def refresh_atm_options_csv(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    out_csv_path: str = "data/atm_options.csv",
) -> None:
    if os.getenv("MARKET_DATA_TOKEN"):
        try:
            update_result = _update_atm_options_csv_from_marketdata(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                out_csv_path=out_csv_path,
            )
            print(
                f"Updated {out_csv_path} from MarketData.app: "
                f"fetched={int(update_result['fetched_rows'])}, total_after_merge={int(update_result['total_rows'])} "
                f"for {len(sorted({str(s).strip() for s in symbols if str(s).strip()}))} symbols "
                f"({start_date}..{end_date})."
            )
        except Exception as exc:
            print(f"Warning: failed to update {out_csv_path} from MarketData.app: {exc}")
    else:
        print("MARKET_DATA_TOKEN not set; skipping MarketData.app ATM options refresh.")


def _resolve_atm_refresh_window(
    symbols: Iterable[str],
    requested_start_date: Optional[str],
    requested_end_date: Optional[str],
    out_csv_path: str,
) -> Tuple[str, str]:
    default_end = (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).date().isoformat()
    end_date = requested_end_date or default_end
    if requested_start_date:
        return requested_start_date, end_date

    default_start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=14)).date().isoformat()
    atm_path = Path(out_csv_path)
    if not atm_path.exists():
        return default_start, end_date

    try:
        existing = pd.read_csv(out_csv_path)
    except Exception:
        return default_start, end_date

    if existing.empty or "date" not in existing.columns:
        return default_start, end_date

    normalized_symbols = {str(s).strip().upper() for s in symbols if str(s).strip()}
    if normalized_symbols and "symbol" in existing.columns:
        existing = existing[existing["symbol"].astype(str).str.upper().isin(normalized_symbols)]

    if existing.empty:
        return default_start, end_date

    dates = pd.to_datetime(existing["date"], errors="coerce").dropna()
    if dates.empty:
        return default_start, end_date

    latest_existing = dates.max().normalize()
    incremental_start = (latest_existing + pd.Timedelta(days=1)).date().isoformat()
    if pd.to_datetime(incremental_start) > pd.to_datetime(end_date):
        incremental_start = end_date
    return incremental_start, end_date


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily signal notifier: checks weekly reversal signal and prints put/call entry notification with estimated ATM premium."
    )
    parser.add_argument(
        "--config",
        default="data/option_signal_notifier.yml",
        help="YAML config file containing strategy definitions.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Optional local CSV path. If omitted, the script prefers data/etfs.csv cache and refreshes only missing rows "
            "from marketstack when cache is stale."
        ),
    )
    parser.add_argument("--start-date", default=None, help="Optional date filter, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional date filter, YYYY-MM-DD.")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk-free rate for Black-Scholes.")
    parser.add_argument("--min-pricing-vol", type=float, default=0.10, help="Vol floor (annualized) used in option pricing.")
    parser.add_argument("--contract-size", type=int, default=100, help="Shares per option contract.")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write machine-readable run summary JSON (keeps stdout logs unchanged).",
    )
    parser.add_argument(
        "--print-trades",
        type=int,
        default=0,
        help="Print recent rows with computed signals in single-config mode. Use -1 for all rows, 0 to disable.",
    )
    args = parser.parse_args()

    configs = _load_configs(args)
    unique_symbols = sorted({cfg.symbol for cfg in configs})

    # Load each symbol once so we refresh cache and fetch API data minimally.
    symbol_data: Dict[str, pd.DataFrame] = {}
    for symbol in unique_symbols:
        symbol_data[symbol] = load_symbol_data(symbol, args.start_date, args.end_date, args.csv)

    atm_start_date, atm_end_date = _resolve_atm_refresh_window(
        symbols=unique_symbols,
        requested_start_date=args.start_date,
        requested_end_date=args.end_date,
        out_csv_path="data/atm_options.csv",
    )
    refresh_atm_options_csv(
        symbols=unique_symbols,
        start_date=atm_start_date,
        end_date=atm_end_date,
        out_csv_path="data/atm_options.csv",
    )

    all_messages: List[str] = []
    config_summaries: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        df = symbol_data[cfg.symbol]
        header = f"\n=== Config #{idx}: {cfg.name} ({cfg.symbol} {cfg.side}) ==="
        all_messages.append(header)
        evaluation = _evaluate_config(
            cfg=cfg,
            symbol_df=df,
            risk_free_rate=args.risk_free_rate,
            min_pricing_vol=args.min_pricing_vol,
            contract_size=args.contract_size,
        )
        if cfg.note and evaluation["fired_signals"]:
            all_messages.append(f"{cfg.name}: {cfg.note}")
        all_messages.extend(evaluation["messages"])
        config_summaries.append({"config_index": idx, **evaluation})

    print("\n".join(all_messages).lstrip())

    if args.summary_json:
        ordered_config_summaries = sorted(
            config_summaries,
            key=lambda cfg: (
                not bool(cfg.get("call_trigger", False)),
                int(cfg.get("config_index", 0)),
            ),
        )
        total_signals = sum(len(cfg["fired_signals"]) for cfg in config_summaries)
        summary_payload = {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "config_count": len(config_summaries),
            "signals_fired": total_signals > 0,
            "signal_count": total_signals,
            "configs": ordered_config_summaries,
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2))
        print(f"Wrote structured signal summary to {summary_path}.")

    if args.print_trades != 0 and len(configs) == 1:
        cfg = configs[0]
        if cfg.signal_model == "accel":
            signal_df = build_acceleration_signal_frame(symbol_data[cfg.symbol], cfg.accel_window, cfg.vol_window)
        elif cfg.signal_model == "roc":
            signal_df = build_signal_frame(symbol_data[cfg.symbol], cfg.roc_lookback, cfg.vol_window)
        else:
            signal_df = build_roc_accel_signal_frame(symbol_data[cfg.symbol], cfg.roc_lookback, cfg.accel_window)
        printable = signal_df.copy()
        if cfg.signal_model == "accel":
            printable["put_signal"] = (
                printable["acceleration"].map(lambda v: _compare(v, cfg.accel_comparator, cfg.accel_threshold))
                & printable["downside_vol_annualized"].map(lambda v: _compare(v, cfg.vol_comparator, cfg.vol_threshold))
            )
            printable["call_signal"] = (
                printable["acceleration"].map(lambda v: _compare(v, cfg.accel_comparator, cfg.accel_threshold))
                & printable["upside_vol_annualized"].map(lambda v: _compare(v, cfg.vol_comparator, cfg.vol_threshold))
            )
            signal_col = "acceleration"
            cols = [
                "date",
                "close",
                signal_col,
                "downside_vol_annualized",
                "upside_vol_annualized",
                "pricing_vol_annualized",
                "put_signal",
                "call_signal",
            ]
        elif cfg.signal_model == "roc":
            printable["put_signal"] = (
                printable["roc"].map(lambda v: _compare(v, cfg.roc_comparator, cfg.roc_threshold))
                & printable["downside_vol_annualized"].map(lambda v: _compare(v, cfg.vol_comparator, cfg.vol_threshold))
            )
            printable["call_signal"] = (
                printable["roc"].map(lambda v: _compare(v, cfg.roc_comparator, cfg.roc_threshold))
                & printable["upside_vol_annualized"].map(lambda v: _compare(v, cfg.vol_comparator, cfg.vol_threshold))
            )
            signal_col = "roc"
            cols = [
                "date",
                "close",
                signal_col,
                "downside_vol_annualized",
                "upside_vol_annualized",
                "pricing_vol_annualized",
                "put_signal",
                "call_signal",
            ]
        else:
            printable["put_signal"] = (
                printable["roc"].map(lambda v: _compare(v, cfg.roc_comparator, cfg.roc_threshold))
                & printable["acceleration"].map(lambda v: _compare(v, cfg.accel_comparator, cfg.accel_threshold))
            )
            printable["call_signal"] = (
                printable["roc"].map(lambda v: _compare(v, cfg.roc_comparator, cfg.roc_threshold))
                & printable["acceleration"].map(lambda v: _compare(v, cfg.accel_comparator, cfg.accel_threshold))
            )
            cols = [
                "date",
                "close",
                "roc",
                "acceleration",
                "pricing_vol_annualized",
                "put_signal",
                "call_signal",
            ]
        to_show = printable[cols] if args.print_trades < 0 else printable[cols].tail(args.print_trades)
        print("\nSignal history:")
        print(to_show.to_string(index=False, justify="center"))
    elif args.print_trades != 0:
        print("\n--print-trades is supported only with a single config row. Ignored for multi-config runs.")


if __name__ == "__main__":
    main()
