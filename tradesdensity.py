import argparse
import math
from datetime import datetime, timezone
from clickhouse_connect import get_client
import pandas as pd
import logging

import multiprocessing as mp
from functools import partial

def get_trades_density(table_name, start_date, end_date, interval, min_density):
    client = get_client()

    query = f"""
    SELECT
        toStartOfInterval(timestamp, INTERVAL {interval} SECOND) AS timestamp,
        toFloat64(any(price)) as open,
        toFloat64(anyLast(price)) as close,
        (toFloat64(anyLast(price)) - toFloat64(any(price))) / toFloat64(any(price)) AS roc,
        count() as count,
        if(
            anyLast(price) = any(price),
            0,
            log(count() / (abs(toFloat64(anyLast(price)) - toFloat64(any(price))) / toFloat64(any(price))))
        ) AS density
    FROM {table_name}
    WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY timestamp
    ORDER BY timestamp ASC
    """
    logger.debug(query)
    result_df = client.query_df(query)
    
    result = []
    for _, row in result_df.iterrows():
        if row['density'] < min_density:
            continue

        result.append((row['timestamp'].replace(tzinfo=timezone.utc), row['open'], row['density']))
        logger.info(f"{row['timestamp'].replace(tzinfo=timezone.utc).isoformat()},{datetime.now(timezone.utc).isoformat()},{row['open']:.5f},{row['density']:.2f}")

    return result


def filter_trade_density(trade_density_list, price_diff_threshold=0.01):
    # Sort by close_price
    trade_density_list.sort(key=lambda x: x[1])
    
    filtered_list = []
    last_kept_close_price = None

    for density_time, close_price, density in trade_density_list:
        if last_kept_close_price is None or abs(close_price - last_kept_close_price) / last_kept_close_price >= price_diff_threshold:
            filtered_list.append((density_time, close_price, density))
            last_kept_close_price = close_price
        else:
            # If the price difference is less than the threshold, keep the one with the higher density
            if density > filtered_list[-1][2]:
                filtered_list[-1] = (density_time, close_price, density)
                last_kept_close_price = close_price

    return filtered_list

def main():
    parser = argparse.ArgumentParser(description="Print prices with unusually low trade density.")
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbol to process, e.g., BTC')
    parser.add_argument('--start_date', default=datetime.strptime("2021-01-01", "%Y-%m-%d"), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', default=datetime.now().replace(microsecond=0), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='End date in YYYY-MM-DD format')
    parser.add_argument('--interval', type=float, default=5*60, help='Set the interval in seconds')
    parser.add_argument('--min_density', type=float, default=18, help='Set the minimum trades density to show')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    args = parser.parse_args()

    logging.basicConfig(format='%(message)s')
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())

    # Process data in chunks and calculate trade density
    all_trade_density = get_trades_density(f"trades_{args.symbol}", args.start_date, args.end_date, args.interval, args.min_density)

    logger.info("Filtering")
    all_trade_density = filter_trade_density(all_trade_density)
    current_date = datetime.now(timezone.utc).isoformat()
    for density_time, close_price, density in all_trade_density:
        logger.info(f"{density_time.replace(tzinfo=timezone.utc).isoformat()},{current_date},{close_price:.5f},{density:.2f}")

if __name__ == "__main__":
    main()



