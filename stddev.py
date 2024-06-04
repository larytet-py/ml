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
        logger.debug(f"{timestamp.isoformat()},{datetime.now(timezone.utc).isoformat()},{row['price']:.5f},{row['rolling_stddev']:.5f}")

    return result

def sum_consolidation_durations(low_stddev_list, price_diff_threshold):
    # Sort the list by price first
    low_stddev_list.sort(key=lambda x: x[1])
    
    # Dictionary to accumulate consolidation durations and start times for each price level
    consolidations = {}
    last_kept_price = None
    current_consolidation_start = None

    for time, price, stddev in low_stddev_list:
        # Check if the price difference is greater than the threshold
        if last_kept_price is None or abs(price - last_kept_price) / last_kept_price >= price_diff_threshold:
            
            # Finalize the current consolidation
            if last_kept_price is not None and current_consolidation_start is not None:
                duration = (time - current_consolidation_start).total_seconds()
                if last_kept_price not in consolidations:
                    consolidations[last_kept_price] = (duration, current_consolidation_start)
                else:
                    consolidations[last_kept_price] = (consolidations[last_kept_price][0] + duration, consolidations[last_kept_price][1])

            # Setup a new consolidation period
            last_kept_price = price
            current_consolidation_start = time
        else:
            # Continue the current consolidation period
            if current_consolidation_start is None:
                current_consolidation_start = time
    
    # Handle the final consolidation period
    if last_kept_price is not None and current_consolidation_start is not None:
        duration = (low_stddev_list[-1][0] - current_consolidation_start).total_seconds()
        if last_kept_price not in consolidations:
            consolidations[last_kept_price] = (duration, current_consolidation_start)
        else:
            consolidations[last_kept_price] = (consolidations[last_kept_price][0] + duration, consolidations[last_kept_price][1])

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

    low_stddev_areas = get_low_stddev_areas(f"trades_{args.symbol}", args.start_date, args.end_date, args.interval, args.max_stddev)

    logger.debug("Filtering")
    consolidations = sum_consolidation_durations(low_stddev_areas, args.price_diff_threshold)
    current_date = datetime.now(timezone.utc).isoformat()
    for price, (duration, time) in consolidations:
        logger.info(f"{time.isoformat()},{current_date},{price:.5f},{duration:.20f}")

if __name__ == "__main__":
    main()
