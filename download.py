import requests
import zipfile
import os
import sys
from datetime import datetime, timedelta
import csv
from concurrent.futures import ThreadPoolExecutor
from clickhouse_connect import get_client
import logging

# Setup the logging configuration
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def check_and_insert_data(csv_path, table_name):
    client = get_client()
    column_names = ['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker', 'unknown_flag'] 

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if not rows:
            logging.error("No data in CSV.")
            return
        first_timestamp, last_timestamp = rows[0][4], rows[-1][4]

    query = f"SELECT count(*) FROM {table_name} WHERE time IN ({first_timestamp}, {last_timestamp})"
    result = client.query(query)
    count_result = result.result_rows[0][0] if result.result_rows else 0

    if count_result < 2:
        logging.error(f"One or both timestamps ({first_timestamp}, {last_timestamp}) are missing in {table_name} for {csv_path}. Inserting data.")
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
            if data:
                logging.debug("Column names: %s", column_names)
                logging.debug("First row of data: %s", data[0])
                client.insert(table=table_name, data=data, column_names=column_names)
        logging.info(f"Data from {csv_path} inserted.")
    else:
        logging.info("Both timestamps are present. No insertion needed.")

def download_and_unpack(url, file_name, csv_name, table_name):
    csv_path = f"./data/{csv_name}"
    if os.path.exists(csv_path):
        logging.info(f"{csv_name} already exists. Checking and updating ClickHouse if necessary.")
        check_and_insert_data(csv_path, table_name)
        return

    logging.info(f"Downloading {file_name}...")
    response = requests.get(url)
    if response.status_code == 200:
        target_path = f"./data/{file_name}"
        with open(target_path, "wb") as f:
            f.write(response.content)

        logging.info(f"Unpacking {file_name}...")
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

if __name__ == "__main__":
    symbol = "BTC"
    start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    end_date = datetime.now()
    num_workers = 1
    table_name = 'trades'

    args = sys.argv[1:]
    if len(args) >= 1:
        symbol = args[0]
    if len(args) >= 3:
        start_date = datetime.strptime(args[1], "%Y-%m-%d")
        end_date = datetime.strptime(args[2], "%Y-%m-%d")
    if len(args) >= 4:
        num_workers = int(args[3])
    if len(args) >= 5:
        log_level = args[4].upper()
        if log_level in ['DEBUG', 'ERROR', 'NONE']:
            logging.getLogger().setLevel(getattr(logging, log_level))

    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    download_files(symbol, start_date, end_date, num_workers, table_name)
