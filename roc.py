import argparse
from datetime import datetime, timezone, timedelta
from clickhouse_connect import get_client
import pandas as pd
import logging

def get_low_volatility_areas(table_name, start_date, end_date, max_roc):
    client = get_client()

    query = f"""
    WITH
        candles AS (
            SELECT
                toUnixTimestamp64Milli(timestamp) AS time,
                open_price,
                close_price,
                (toFloat64(close_price) - toFloat64(open_price)) / toFloat64(open_price) AS roc
            FROM {table_name}
            WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        )
    SELECT
        time,
        roc,
        toFloat64(open_price) AS open_price,
        toFloat64(close_price) AS close_price
    FROM candles
    WHERE roc < {max_roc}
    ORDER BY time ASC
    """
    logger.debug(query)
    result_df = client.query_df(query)

    return result_df

def sum_consolidation_durations(df, price_diff_threshold):
    consolidations = {}

    for _, row in df.iterrows():
        time = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=row['time'])
        open_price = row['open_price']
        rounded_price = round(open_price / (10 * price_diff_threshold)) * (10 * price_diff_threshold)

        stored_value = consolidations.get(rounded_price, (0, None))
        consolidation_start = time if stored_value[1] is None else stored_value[1]
        duration = (time - consolidation_start).total_seconds()

        consolidations[rounded_price] = (stored_value[0] + duration, consolidation_start)

    return consolidations

def filter_consolidations(consolidations, price_diff_threshold):
    # Convert the dictionary to a list of tuples for easier sorting and filtering
    consolidation_list = [(price, duration, start_time) for price, (duration, start_time) in consolidations.items()]

    # Sort by price
    consolidation_list.sort(key=lambda x: x[0])
    
    filtered_list = []
    last_kept_price = None

    for price, duration, start_time in consolidation_list:
        if last_kept_price is None or abs(price - last_kept_price) / last_kept_price >= price_diff_threshold:
            filtered_list.append((price, duration, start_time))
            last_kept_price = price
        else:
            # If the price difference is less than the threshold, keep the one with the longer duration
            if duration > filtered_list[-1][1]:
                filtered_list[-1] = (price, duration, start_time)
                last_kept_price = price

    return filtered_list

def main():
    parser = argparse.ArgumentParser(description="Print price levels with longest consolidations by time.")
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbol to process, e.g., BTC')
    parser.add_argument('--start_date', default=datetime.strptime("2021-01-01", "%Y-%m-%d"), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', default=datetime.now().replace(microsecond=0), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='End date in YYYY-MM-DD format')
    parser.add_argument('--max_roc', type=float, default=0.004, help='Set the maximum ROC to show')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    parser.add_argument('--price_diff_threshold', type=float, default=0.005, help='Set the minimum distance between the price levels')
    args = parser.parse_args()

    logging.basicConfig(format='%(message)s')
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())

    logger.info("Collect")
    low_volatility_df = get_low_volatility_areas(f"ohlc_M1_{args.symbol}", args.start_date, args.end_date, args.max_roc)

    logger.info(f"Get durations for {len(low_volatility_df)} periods")
    consolidations = sum_consolidation_durations(low_volatility_df, args.price_diff_threshold)

    logger.info("Filter")
    consolidations = filter_consolidations(consolidations, args.price_diff_threshold)
    current_date = datetime.now(timezone.utc).isoformat()
    consolidations.sort(key=lambda x: x[0])
    logger.info(f"Got {len(consolidations)}")
    for price, duration, time in consolidations:
        logger.info(f"{time.isoformat()},{current_date},{price:.2f},{duration:.0f}")

if __name__ == "__main__":
    main()
