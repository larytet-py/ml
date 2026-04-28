#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

import requests


DEFAULT_BASE_URL = "https://api.massive.com"
DEFAULT_SYMBOLS = ["QQQ", "IWM", "VXX", "TLT", "IBIT", "GDX", "SPY"]
DEFAULT_START_DATE = "2024-04-24"
DEFAULT_END_DATE = date.today().isoformat()
DEFAULT_CONTRACT_TYPES = ["call", "put"]

FIELDNAMES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ATM option OHLC bars from Polygon/Massive for a list of ETFs."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols, e.g. QQQ,IWM,VXX,TLT,IBIT,GDX,SPY",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--contract-types",
        default=",".join(DEFAULT_CONTRACT_TYPES),
        help="Comma-separated contract types: call,put",
    )
    parser.add_argument(
        "--expiry-lookahead-days",
        type=int,
        default=14,
        help="Find nearest expiration from trade date through trade date + this many days.",
    )
    parser.add_argument(
        "--strike-window-pct",
        type=float,
        default=0.10,
        help="Strike search window as percentage of underlying close.",
    )
    parser.add_argument(
        "--min-strike-window",
        type=float,
        default=10.0,
        help="Minimum absolute strike search window.",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=1.0,
        help="Delay before each request. Increase this if you hit 429 rate limits.",
    )
    parser.add_argument(
        "--out",
        default="atm_option_ohlc.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API base URL. Default: https://api.massive.com",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print contract query counts and missing reasons.",
    )
    parser.add_argument(
        "--api-key-env",
        default="POLYGON_API_KEY",
        help="Environment variable containing API key. Default: POLYGON_API_KEY",
    )
    return parser.parse_args()


class MassiveClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        request_sleep: float,
        debug: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_sleep = request_sleep
        self.debug = debug
        self.session = requests.Session()

    def get_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None}
        params["apiKey"] = self.api_key
        url = f"{self.base_url}{path}?{urlencode(params)}"

        for attempt in range(1, 8):
            time.sleep(self.request_sleep)

            try:
                response = self.session.get(url, timeout=30)
            except requests.RequestException as exc:
                wait = min(60, 2 ** attempt)
                print(f"Request error attempt={attempt}: {exc}. Sleeping {wait}s.", file=sys.stderr)
                time.sleep(wait)
                continue

            if response.status_code == 429:
                wait = min(120, 2 ** attempt)
                print(f"Rate limited 429. Sleeping {wait}s.", file=sys.stderr)
                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:
                wait = min(60, 2 ** attempt)
                print(f"Server error {response.status_code}. Sleeping {wait}s.", file=sys.stderr)
                time.sleep(wait)
                continue

            response.raise_for_status()
            payload = response.json()

            if self.debug and path == "/v3/reference/options/contracts":
                safe_url = url.replace(self.api_key, "API_KEY")
                print(f"contracts url: {safe_url}")
                print(f"contracts count: {len(payload.get('results', []) or [])}")

            return payload

        raise RuntimeError(f"Failed after retries: {url.replace(self.api_key, 'API_KEY')}")

    def iter_paginated(self, path: str, params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        payload = self.get_json(path, params)

        while True:
            for row in payload.get("results", []) or []:
                yield row

            next_url = payload.get("next_url")
            if not next_url:
                break

            if "apiKey=" not in next_url:
                sep = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{sep}apiKey={self.api_key}"

            for attempt in range(1, 8):
                time.sleep(self.request_sleep)

                try:
                    response = self.session.get(next_url, timeout=30)
                except requests.RequestException as exc:
                    wait = min(60, 2 ** attempt)
                    print(
                        f"Pagination request error attempt={attempt}: {exc}. Sleeping {wait}s.",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue

                if response.status_code == 429:
                    wait = min(120, 2 ** attempt)
                    print(f"Pagination rate limited 429. Sleeping {wait}s.", file=sys.stderr)
                    time.sleep(wait)
                    continue

                if 500 <= response.status_code < 600:
                    wait = min(60, 2 ** attempt)
                    print(
                        f"Pagination server error {response.status_code}. Sleeping {wait}s.",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                payload = response.json()
                break
            else:
                raise RuntimeError("Pagination failed after retries.")


def ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def fetch_underlying_daily(
    client: MassiveClient,
    symbol: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"

    payload = client.get_json(
        path,
        {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        },
    )

    rows: List[Dict[str, Any]] = []
    for r in payload.get("results", []) or []:
        if "t" not in r:
            continue

        rows.append(
            {
                "symbol": symbol,
                "date": ms_to_date(r["t"]),
                "underlying_open": r.get("o"),
                "underlying_high": r.get("h"),
                "underlying_low": r.get("l"),
                "underlying_close": r.get("c"),
                "underlying_volume": r.get("v"),
            }
        )

    return rows


def is_past_trade_date(trade_date: str) -> bool:
    return date.fromisoformat(trade_date) < date.today()


def fetch_contracts(
    client: MassiveClient,
    symbol: str,
    contract_type: str,
    trade_date: str,
    underlying_close: float,
    expiry_lookahead_days: int,
    strike_window_pct: float,
    min_strike_window: float,
) -> List[Dict[str, Any]]:
    d = date.fromisoformat(trade_date)
    expiration_to = (d + timedelta(days=expiry_lookahead_days)).isoformat()

    strike_window = max(min_strike_window, underlying_close * strike_window_pct)
    strike_min = round(underlying_close - strike_window, 2)
    strike_max = round(underlying_close + strike_window, 2)

    # Important:
    # Do NOT pass as_of here for historical lookups.
    # For old dates, use expired=true.
    expired = "true" if is_past_trade_date(trade_date) else "false"

    return list(
        client.iter_paginated(
            "/v3/reference/options/contracts",
            {
                "underlying_ticker": symbol,
                "contract_type": contract_type,
                "expiration_date.gte": trade_date,
                "expiration_date.lte": expiration_to,
                "strike_price.gte": strike_min,
                "strike_price.lte": strike_max,
                "expired": expired,
                "limit": 1000,
                "sort": "expiration_date",
                "order": "asc",
            },
        )
    )


def choose_atm_contract(
    contracts: List[Dict[str, Any]],
    underlying_close: float,
) -> Optional[Dict[str, Any]]:
    if not contracts:
        return None

    def sort_key(contract: Dict[str, Any]) -> Any:
        expiration = contract.get("expiration_date", "9999-12-31")
        strike = float(contract.get("strike_price", 0.0))
        ticker = contract.get("ticker", "")
        return (
            expiration,
            abs(strike - underlying_close),
            strike,
            ticker,
        )

    return min(contracts, key=sort_key)


def fetch_option_daily_bar(
    client: MassiveClient,
    option_ticker: str,
    trade_date: str,
) -> Optional[Dict[str, Any]]:
    path = f"/v2/aggs/ticker/{option_ticker}/range/1/day/{trade_date}/{trade_date}"

    payload = client.get_json(
        path,
        {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        },
    )

    results = payload.get("results", []) or []
    if not results:
        return None

    return results[0]


def empty_option_row(
    underlying_row: Dict[str, Any],
    contract_type: str,
    missing_reason: str,
) -> Dict[str, Any]:
    return {
        **underlying_row,
        "contract_type": contract_type,
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
        "missing_reason": missing_reason,
    }


def fetch_one_atm_option_row(
    client: MassiveClient,
    symbol: str,
    underlying_row: Dict[str, Any],
    contract_type: str,
    expiry_lookahead_days: int,
    strike_window_pct: float,
    min_strike_window: float,
) -> Dict[str, Any]:
    trade_date = underlying_row["date"]

    try:
        if underlying_row["underlying_close"] is None:
            return empty_option_row(underlying_row, contract_type, "missing_underlying_close")

        close = float(underlying_row["underlying_close"])

        contracts = fetch_contracts(
            client=client,
            symbol=symbol,
            contract_type=contract_type,
            trade_date=trade_date,
            underlying_close=close,
            expiry_lookahead_days=expiry_lookahead_days,
            strike_window_pct=strike_window_pct,
            min_strike_window=min_strike_window,
        )

        contract = choose_atm_contract(contracts, close)
        if not contract:
            return empty_option_row(underlying_row, contract_type, "no_contract_found")

        option_ticker = contract["ticker"]
        bar = fetch_option_daily_bar(client, option_ticker, trade_date)

        if bar is None:
            return {
                **underlying_row,
                "contract_type": contract_type,
                "option_ticker": option_ticker,
                "expiration_date": contract.get("expiration_date"),
                "strike_price": contract.get("strike_price"),
                "option_open": None,
                "option_high": None,
                "option_low": None,
                "option_close": None,
                "option_volume": None,
                "option_vwap": None,
                "option_transactions": None,
                "missing_reason": "no_option_bar",
            }

        return {
            **underlying_row,
            "contract_type": contract_type,
            "option_ticker": option_ticker,
            "expiration_date": contract.get("expiration_date"),
            "strike_price": contract.get("strike_price"),
            "option_open": bar.get("o"),
            "option_high": bar.get("h"),
            "option_low": bar.get("l"),
            "option_close": bar.get("c"),
            "option_volume": bar.get("v"),
            "option_vwap": bar.get("vw"),
            "option_transactions": bar.get("n"),
            "missing_reason": None,
        }

    except Exception as exc:
        return empty_option_row(
            underlying_row,
            contract_type,
            f"error: {type(exc).__name__}: {exc}",
        )


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    contract_types = [s.strip().lower() for s in args.contract_types.split(",") if s.strip()]

    bad_contract_types = [ct for ct in contract_types if ct not in {"call", "put"}]
    if bad_contract_types:
        raise ValueError(f"Invalid contract types: {bad_contract_types}. Use call,put.")

    client = MassiveClient(
        api_key=api_key,
        base_url=args.base_url,
        request_sleep=args.request_sleep,
        debug=args.debug,
    )

    all_underlying_rows: Dict[str, List[Dict[str, Any]]] = {}

    for symbol in symbols:
        print(f"Fetching underlying daily bars for {symbol}...")
        rows = fetch_underlying_daily(client, symbol, args.start_date, args.end_date)
        print(f"{symbol}: {len(rows)} underlying bars")
        all_underlying_rows[symbol] = rows

    output_rows: List[Dict[str, Any]] = []

    jobs = []
    for symbol, underlying_rows in all_underlying_rows.items():
        for underlying_row in underlying_rows:
            for contract_type in contract_types:
                jobs.append((symbol, underlying_row, contract_type))

    total = len(jobs)
    print(f"Submitted {total} option jobs sequentially.")

    for i, (symbol, underlying_row, contract_type) in enumerate(jobs, start=1):
        row = fetch_one_atm_option_row(
            client=client,
            symbol=symbol,
            underlying_row=underlying_row,
            contract_type=contract_type,
            expiry_lookahead_days=args.expiry_lookahead_days,
            strike_window_pct=args.strike_window_pct,
            min_strike_window=args.min_strike_window,
        )
        output_rows.append(row)

        if args.debug and row.get("missing_reason"):
            print(
                f"missing: {row['date']} {row['symbol']} {contract_type}: "
                f"{row['missing_reason']}"
            )

        if i % 10 == 0 or i == total:
            missing = sum(1 for r in output_rows if r.get("missing_reason"))
            print(f"Completed {i}/{total}; missing/error rows so far: {missing}")

    output_rows.sort(
        key=lambda r: (
            str(r.get("symbol") or ""),
            str(r.get("date") or ""),
            str(r.get("contract_type") or ""),
            str(r.get("expiration_date") or ""),
            float(r.get("strike_price") or 0),
        )
    )

    write_csv(args.out, output_rows)

    missing_counts: Dict[str, int] = {}
    for row in output_rows:
        reason = row.get("missing_reason")
        if reason:
            missing_counts[reason] = missing_counts.get(reason, 0) + 1

    print(f"Wrote {len(output_rows)} rows to {args.out}")

    if missing_counts:
        print("Missing/error summary:")
        for reason, count in sorted(missing_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
