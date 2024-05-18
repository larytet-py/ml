import argparse
import math
from datetime import datetime, timezone
from clickhouse_connect import get_client
import pandas as pd
import logging

# Function to calculate ROC and trade density for each chunk
def calculate_metrics(df, interval='5S'):
    all_trade_density = []
    # Convert timestamp to pandas datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Group by specified intervals
    grouped = df.groupby(df['timestamp'].dt.floor(interval))

    for timestamp, group in grouped:
        if len(group) > 1:
            open_price = group.iloc[0]['price']
            close_price = group.iloc[-1]['price']
            roc = math.fabs((close_price - open_price) / open_price)

            trade_count = len(group)
            if roc != 0:  # Avoid division by zero
                trade_density = trade_count / roc
                all_trade_density.append((trade_density, timestamp, open_price, close_price))

    return all_trade_density

# Function to process data in chunks, calculate trade density
def process_data_in_chunks(query, chunk_size, interval, min_density):
    current_date = datetime.now(timezone.utc).isoformat()
    offset = 0
    all_trade_density = []

    client = get_client()

    while True:
        chunk_query = query.format(limit=chunk_size, offset=offset)
        chunk_df = client.query_df(chunk_query)
        
        if chunk_df.empty:
            break

        # Calculate metrics for the current chunk
        chunk_trade_density = calculate_metrics(chunk_df, interval)
        # all_trade_density.extend(chunk_trade_density)

        for density, density_time, _, close_price in chunk_trade_density:
            if density > min_density:
                all_trade_density.append((density_time.replace(tzinfo=timezone.utc), close_price, density))


        if chunk_trade_density:
            #min_density_record = min(chunk_trade_density, key=lambda x: x[0])
            max_density_record = max(chunk_trade_density, key=lambda x: x[0])
            #min_density, min_density_time, min_open_price, min_close_price = min_density_record
            max_density, max_density_time, max_open_price, max_close_price = max_density_record
            max_log_density = math.log(max_density)
            #chunk_timestamp = chunk_df['timestamp'].iloc[0]
            if max_log_density > 22:
                # logger.debug(f"{chunk_timestamp}: "
                #             f"{min_log_density:.2f}@{min_density_time}, price={min_open_price:.0f}/{min_close_price:.0f},"
                #             f"{max_log_density:.2f}@{max_density_time}, price={max_open_price:.0f}/{max_close_price:.0f}"
                # )
                # print(f"{max_density_time.isoformat()},{current_date},{max_close_price:.5f}")
                print(f"{max_density_time.replace(tzinfo=timezone.utc).isoformat()},{current_date},{max_close_price:.5f},{max_log_density:.2f}")

        # Increment offset for the next chunk
        offset += chunk_size


    return all_trade_density

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
    parser.add_argument('--start_date', default=datetime.strptime("2024-01-01", "%Y-%m-%d"), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', default=datetime.now(), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='End date in YYYY-MM-DD format')
    parser.add_argument('--interval', type=float, default=5.0, help='Set the interval in seconds')
    parser.add_argument('--min_density', type=float, default=0.5, help='Set the minimum trades density to show')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='DEBUG', help='Set the logging level')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())

    query_template = f"""
    SELECT id, price, qty, base_qty, time, is_buyer_maker, unknown_flag, timestamp
    FROM trades_{args.symbol}
    WHERE timestamp BETWEEN '{args.start_date}' AND '{args.end_date}'
    ORDER BY timestamp
    LIMIT {{limit}} OFFSET {{offset}}
    """

    chunk_size = 1_000_000
    interval_str = f'{int(args.interval)}S'

    # Process data in chunks and calculate trade density
    all_trade_density = process_data_in_chunks(query_template, chunk_size, interval_str, 22)

    all_trade_density = filter_trade_density(all_trade_density)
    print(all_trade_density)

if __name__ == "__main__":
    main()



