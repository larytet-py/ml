import argparse
from datetime import datetime, timezone
from clickhouse_connect import get_client
import pandas as pd
import logging

def get_low_stddev_areas(table_name, start_date, end_date, interval, max_stddev):
    client = get_client()

    query = f"""
    WITH
        trades AS (
            SELECT
                toUnixTimestamp64Milli(timestamp) AS time,
                price,
                1 AS trade_count
            FROM {table_name}
            WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        ),
        aggregated_trades AS (
            SELECT
                intDiv(time, 10) * 10 AS time_bucket, -- aggregate the trades into 10-millisecond buckets
                avg(price) AS avg_price
            FROM trades
            GROUP BY time_bucket
        ),
        rolling_metrics AS (
            SELECT
                time_bucket,
                stddevPop(avg_price) OVER (
                    ORDER BY time_bucket
                    ROWS BETWEEN intDiv({interval}, 10) PRECEDING AND CURRENT ROW -- calculate rolling stddev for the period
                ) AS rolling_stddev_price
            FROM aggregated_trades
        )
    SELECT
        time_bucket AS time,
        rolling_stddev_price AS rolling_stddev,
        avg_price AS price
    FROM rolling_metrics
    WHERE rolling_stddev_price < {max_stddev}
    ORDER BY time ASC
    """
    logger.debug(query)
    result_df = client.query_df(query)
    
    result = []
    for _, row in result_df.iterrows():
        result.append((row['time'].replace(tzinfo=timezone.utc), row['price'], row['rolling_stddev']))
        logger.debug(f"{row['time'].replace(tzinfo=timezone.utc).isoformat()},{datetime.now(timezone.utc).isoformat()},{row['price']:.5f},{row['rolling_stddev']:.5f}")

    return result


def filter_low_stddev_areas(low_stddev_list, price_diff_threshold):
    low_stddev_list.sort(key=lambda x: x[1])
    
    filtered_list = []
    last_kept_price = None

    for time, price, stddev in low_stddev_list:
        if last_kept_price is None or abs(price - last_kept_price) / last_kept_price >= price_diff_threshold:
            filtered_list.append((time, price, stddev))
            last_kept_price = price
        else:
            if stddev < filtered_list[-1][2]:
                filtered_list[-1] = (time, price, stddev)
                last_kept_price = price

    return filtered_list

def main():
    parser = argparse.ArgumentParser(description="Print price levels with longest consolidations by time.")
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbol to process, e.g., BTC')
    parser.add_argument('--start_date', default=datetime.strptime("2021-01-01", "%Y-%m-%d"), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', default=datetime.now().replace(microsecond=0), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='End date in YYYY-MM-DD format')
    parser.add_argument('--interval', type=float, default=60, help='Set the interval in seconds')
    parser.add_argument('--max_stddev', type=float, default=0.01, help='Set the maximum standard deviation to show')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    parser.add_argument('--price_diff_threshold', type=float, default=0.01, help='Set the minimum distance between the price levels')
    args = parser.parse_args()

    logging.basicConfig(format='%(message)s')
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())

    low_stddev_areas = get_low_stddev_areas(f"trades_{args.symbol}", args.start_date, args.end_date, args.interval, args.max_stddev)

    logger.debug("Filtering")
    filtered_areas = filter_low_stddev_areas(low_stddev_areas, args.price_diff_threshold)
    current_date = datetime.now(timezone.utc).isoformat()
    for time, price, stddev in filtered_areas:
        logger.info(f"{time.replace(tzinfo=timezone.utc).isoformat()},{current_date},{price:.5f},{stddev:.5f}")

if __name__ == "__main__":
    main()
