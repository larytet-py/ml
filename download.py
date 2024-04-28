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
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def get_first_and_last_timestamp_with_pandas(csv_name):
    df = pd.read_csv(csv_name, usecols=[4])  # 'time' is in the 5th column (index 4)
    if df.empty:
        return None, None
    first_timestamp = df.iloc[0, 0]
    last_timestamp = df.iloc[-1, 0]
    return first_timestamp, last_timestamp

def check_and_insert_data(csv_name, table_name):
    client = get_client()
    column_names = ['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker', 'unknown_flag'] 
    first_timestamp, last_timestamp = get_first_and_last_timestamp_with_pandas(csv_name)

    query = f"SELECT count(*) FROM {table_name} WHERE time IN ({first_timestamp}, {last_timestamp})"
    result = client.query(query)
    count_result = result.result_rows[0][0] if result.result_rows else 0

    if count_result < 2:
        logging.error(f"One or both timestamps ({first_timestamp}, {last_timestamp}) are missing in {table_name} for {csv_name}. Inserting data.")
        with open(csv_name, 'r') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
            if data:
                client.insert(table=table_name, data=data, column_names=column_names)
        logging.info(f"Data from {csv_name} inserted.")
    else:
        logging.info(f"Timestamps {first_timestamp}, {last_timestamp} are present for {csv_name}. No update is needed.")

def download_and_unpack(url, file_name, csv_name, table_name):
    csv_path = f"./data/{csv_name}"
    if os.path.exists(csv_path):
        logging.debug(f"{csv_name} already exists. Checking and updating ClickHouse if necessary.")
        check_and_insert_data(csv_path, table_name)
        return

    logging.info(f"Downloading {file_name}...")
    response = requests.get(url)
    if response.status_code == 200:
        target_path = f"./data/{file_name}"
        with open(target_path, "wb") as f:
            f.write(response.content)

        logging.debug(f"Unpacking {file_name}...")
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall("./data/")

        os.remove(target_path)
        check_and_insert_data(csv_path, table_name)
    else:
        logging.error(f"Failed to download {file_name} or file does not exist for this date.")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process trade data files.")
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbol to process, e.g., BTC')
    parser.add_argument('--start_date', default=datetime.strptime("2024-01-01", "%Y-%m-%d"), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', default=datetime.now(), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='End date in YYYY-MM-DD format')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes/threads')
    parser.add_argument('--table_name', type=str, default='trades', help='Name of the database table to insert data into')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='ERROR', help='Set the logging level')

    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    download_files_process(symbol=args.symbol, start_date=args.start_date, end_date=args.end_date, num_workers=args.num_workers, table_name=args.table_name)
