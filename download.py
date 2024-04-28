import requests
import zipfile
import os
from datetime import datetime, timedelta
import sys
from concurrent.futures import ThreadPoolExecutor
import csv
from clickhouse_connect import get_client
import time

def check_and_insert_data(csv_path, table_name):
    client = get_client()
    # Read first and last timestamp from the CSV
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        first_timestamp, last_timestamp = rows[0][0], rows[-1][0]

    # Check if these timestamps exist in ClickHouse
    query = f"SELECT count(*) FROM {table_name} WHERE time IN ({first_timestamp}, {last_timestamp})"
    result = client.query(query)
    if result[0][0] < 2:  # If one or both timestamps are missing
        print("Missing timestamps detected. Inserting data.")
        with open(csv_path, 'r') as file:
            client.insert_csv(table_name, file, settings={'async_insert': True})
        
        # Assuming a reasonable delay to wait for async insertion to complete
        print("Waiting for async insertion to complete...")
        time.sleep(10)  # Adjust time based on your observations of typical insert times
        
        # Verify insertion
        verification_query = f"SELECT count(*) FROM {table_name} WHERE time IN ({first_timestamp}, {last_timestamp})"
        verification_result = client.query(verification_query)
        if verification_result[0][0] >= 2:
            print("Data verification successful: Data inserted correctly.")
        else:
            print("Data verification failed: Data might not have been inserted correctly.")
    else:
        print("Timestamps already present. No insertion needed.")


def download_and_unpack(url, file_name, csv_name, table_name):
    csv_path = f"./data/{csv_name}"
    if os.path.exists(csv_path):
        print(f"{csv_name} already exists. Checking and updating ClickHouse if necessary.")
        check_and_insert_data(csv_path, table_name)
        return

    print(f"Downloading {file_name}...")
    response = requests.get(url)
    if response.status_code == 200:
        target_path = f"./data/{file_name}"
        with open(target_path, "wb") as f:
            f.write(response.content)

        print(f"Unpacking {file_name}...")
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall("./data/")

        os.remove(target_path)
        check_and_insert_data(csv_path, table_name)
    else:
        print(f"Failed to download {file_name} or file does not exist for this date.")

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

    if not os.path.exists("./data/"):
        os.makedirs("./data/")


    download_files(symbol, start_date, end_date, num_workers, table_name)