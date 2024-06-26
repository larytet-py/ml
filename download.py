import argparse
import pandas as pd
import requests
import zipfile
import os
import sys
from datetime import datetime, timedelta
import csv
from concurrent.futures import ProcessPoolExecutor
from clickhouse_connect import get_client
import logging

# Setup the logging configuration
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')

def get_first_and_last_timestamp_with_pandas(csv_name):
    df = pd.read_csv(csv_name, usecols=[4])  # 'time' is in the 5th column (index 4)
    if df.empty:
        return None, None
    first_timestamp = df.iloc[0, 0]
    last_timestamp = df.iloc[-1, 0]
    return first_timestamp, last_timestamp

def check_and_insert_data(csv_name, table_name):
    client = get_client(settings={'insert_deduplicate': '1'})

    column_names = ['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker', 'unknown_flag', 'timestamp']
    df = pd.read_csv(csv_name, header=None, names=column_names)

    # Query to check the existence of all timestamps in the table
    first_timestamp, last_timestamp = df['time'].min(), df['time'].max()
    query = f"SELECT count(*) FROM {table_name} WHERE time BETWEEN '{first_timestamp}' AND '{last_timestamp}'"
    result = client.query(query)
    count_result = result.result_rows[0][0] if result.result_rows else 0
    if count_result >= 2:
        logging.info(f"Timestamps between {first_timestamp} and {last_timestamp} are present for {csv_name}. No update is needed.")
        return

    logging.info(f"One or both timestamps ({first_timestamp}, {last_timestamp}) are missing in {table_name} for {csv_name}. Inserting data.")

    # Prepare data for insertion
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')

    # Inserting data from dataframe
    result = client.insert_df(table=table_name, df=df)
    f = {False: logging.error, True: logging.info}[result.written_rows == len(df)]
    f(f"{result.written_rows} of {len(df)} rows from {csv_name} inserted.")

    client.close()

def download_and_unpack(url, file_name, csv_name, table_name):
    csv_path = f"./data/{csv_name}"
    if os.path.exists(csv_path):
        logging.debug(f"{csv_name} already exists. Checking and updating {table_name} if necessary.")
        check_and_insert_data(csv_path, table_name)
        return

    logging.info(f"Downloading {file_name}...")
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to download {file_name} or file does not exist for this date.")
        return

    target_path = f"./data/{file_name}"
    with open(target_path, "wb") as f:
        f.write(response.content)

    logging.debug(f"Unpacking {file_name}...")
    with zipfile.ZipFile(target_path, 'r') as zip_ref:
        zip_ref.extractall("./data/")

    os.remove(target_path)
    check_and_insert_data(csv_path, table_name)
    os.remove(csv_path)
        

def download_files(symbol, start_date, end_date, num_workers, table_name):
    base_url = "https://data.binance.vision/data/spot/daily/trades/{symbol}USDT/{symbol}USDT-trades-{date}.zip"
    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    for current_date in dates:
        date_str = current_date.strftime("%Y-%m-%d")
        url = base_url.format(symbol=symbol, date=date_str)
        file_name = f"{symbol}USDT-trades-{date_str}.zip"
        csv_name = f"{symbol}USDT-trades-{date_str}.csv"
        download_and_unpack(url, file_name, csv_name, table_name)

def download_files_process(symbol, start_date, end_date, num_workers, table_name):
    base_url = "https://data.binance.vision/data/spot/daily/trades/{symbol}USDT/{symbol}USDT-trades-{date}.zip"
    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for current_date in dates:
            date_str = current_date.strftime("%Y-%m-%d")
            url = base_url.format(symbol=symbol, date=date_str)
            file_name = f"{symbol}USDT-trades-{date_str}.zip"
            csv_name = f"{symbol}USDT-trades-{date_str}.csv"
            futures.append(executor.submit(download_and_unpack, url, file_name, csv_name, table_name))



def create_table_trades(table_name):
    check_table_exists_query = f"EXISTS TABLE {table_name}"
    create_table_query = f"""
    CREATE TABLE {table_name}
    (
        id UInt64,
        price Decimal(18, 8),
        qty Decimal(12, 8),
        base_qty Decimal(12, 8),
        time UInt64,
        is_buyer_maker Boolean,
        unknown_flag Boolean DEFAULT True,
        timestamp DateTime64(3) -- Millisecond precision
    )
    ENGINE = MergeTree
    PARTITION BY toDate(toDateTime(time / 1000))
    ORDER BY (timestamp)
    SETTINGS index_granularity = 8192;
    """

    client = get_client()
    result = client.query(check_table_exists_query)
    count_result = result.result_rows[0][0] if result.result_rows else 0
    if count_result > 0:
        logging.debug(f"Table '{table_name}' already exists.")
        return
    
    client.query(create_table_query)
    client.close()
    logging.info(f"Table '{table_name}' created successfully.")

def optimize(table_name):
    logging.info(f"Optimize '{table_name}'.")
    create_table_query = f"""OPTIMIZE {table_name} FINAL"""
    client = get_client()
    client.query(create_table_query)
    client.close()

def fetch_and_aggregate_data(client, table_name, offset=0, batch_size=10000):
    trades_query = f"""
    SELECT
        timestamp,
        CAST(price AS Float64) AS price,
        CAST(base_qty AS Float64) AS base_qty
    FROM {table_name}
    ORDER BY timestamp
    LIMIT {batch_size} OFFSET {offset}    
    """
    df = client.query_df(trades_query)
    
    if df.empty:
        return df  # Return empty DataFrame if no data is fetched

    df.set_index('timestamp', inplace=True)

    # Resample
    ohlc = df['price'].resample('1min').ohlc()
    volume = df['base_qty'].resample('1min').sum()

    # Prepare DataFrame for insertion
    ohlc['volume'] = volume
    ohlc.reset_index(inplace=True)
    ohlc.columns = ['time_start', 'open', 'high', 'low', 'close', 'volume']
    ohlc['time_end'] = ohlc['time_start'] + timedelta(minutes=1)

    # Handling missing data by filling with last known values
    ohlc.fillna(method='ffill', inplace=True)
    
    return ohlc

def insert_data(client, df, table_name):
    client.insert_df(table_name, df)
    readable_time = df['time_end'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')    
    logging.info(f"Data inserted successfully into {table_name} {readable_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process trade data files.")
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbol to process, e.g., BTC')
    parser.add_argument('--start_date', default=datetime.strptime("2021-01-01", "%Y-%m-%d"), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', default=datetime.now(), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='End date in YYYY-MM-DD format')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes/threads')
    parser.add_argument('--disable-download', action='store_true', help='Disable the download functionality.')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    table_name = f"trades_{args.symbol}"
    create_table_trades(table_name)
    if not args.disable_download:
        download_files_process(symbol=args.symbol, start_date=args.start_date, end_date=args.end_date, num_workers=args.num_workers, table_name=table_name)  
