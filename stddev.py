import argparse
from datetime import datetime, timezone, timedelta
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
                intDiv(time, 100) * 100 AS time_bucket, -- aggregate the trades into 100-millisecond buckets
                avg(price) AS avg_price
            FROM trades
            GROUP BY time_bucket
        ),
        rolling_metrics AS (
            SELECT
                time_bucket,
                avg_price,
                stddevPop(avg_price) OVER (
                    ORDER BY time_bucket
                    ROWS BETWEEN intDiv({interval}, 100) PRECEDING AND CURRENT ROW -- calculate rolling stddev for the period
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
        # Convert Unix timestamp in milliseconds to a datetime object with millisecond precision
        timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=row['time'])
        result.append((timestamp, row['price'], row['rolling_stddev']))

    return result

def sum_consolidation_durations(low_stddev_list, price_diff_threshold):
    consolidations = {}

    for time, price, _ in low_stddev_list:
        rounded_price = round(price/(10*price_diff_threshold))*(10*price_diff_threshold)

        stored_value = consolidations.get(rounded_price, (0, None))
        consolidation_start = time if stored_value[1] is None else stored_value[1]
        duration = (time - consolidation_start).total_seconds()

        consolidations[rounded_price] = (stored_value[0] + duration, consolidation_start)

    return consolidations


def main():
    parser = argparse.ArgumentParser(description="Print price levels with longest consolidations by time.")
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbol to process, e.g., BTC')
    parser.add_argument('--start_date', default=datetime.strptime("2021-01-01", "%Y-%m-%d"), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', default=datetime.now().replace(microsecond=0), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='End date in YYYY-MM-DD format')
    parser.add_argument('--interval', type=float, default=5, help='Set the interval in seconds')
    parser.add_argument('--max_stddev', type=float, default=0.004, help='Set the maximum standard deviation to show')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    parser.add_argument('--price_diff_threshold', type=float, default=0.0001, help='Set the minimum distance between the price levels')
    args = parser.parse_args()

    logging.basicConfig(format='%(message)s')
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())

    logger.info("Collect")
    low_stddev_areas = get_low_stddev_areas(f"trades_{args.symbol}", args.start_date, args.end_date, args.interval, args.max_stddev)

    logger.info("Filtering")
    consolidations = sum_consolidation_durations(low_stddev_areas, args.price_diff_threshold)
    current_date = datetime.now(timezone.utc).isoformat()
    for price, (duration, time) in sorted(consolidations.items()):
        logger.info(f"{time.isoformat()},{current_date},{price:.2f},{duration:.0f}")

if __name__ == "__main__":
    main()
