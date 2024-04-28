import requests
import zipfile
import os
from datetime import datetime, timedelta
import sys
from concurrent.futures import ThreadPoolExecutor

def download_and_unpack(url, file_name, csv_name):
    # Check if the CSV file already exists to avoid re-downloading and unpacking
    if os.path.exists(f"./data/{csv_name}"):
        print(f"{csv_name} already exists. Skipping download and unpack.")
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
    else:
        print(f"Failed to download {file_name} or file does not exist for this date.")

def download_files(symbol, start_date, end_date, num_workers):
    base_url = "https://data.binance.vision/data/spot/daily/trades/{symbol}USDT/{symbol}USDT-trades-{date}.zip"
    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for current_date in dates:
            date_str = current_date.strftime("%Y-%m-%d")
            url = base_url.format(symbol=symbol, date=date_str)
            file_name = f"{symbol}USDT-trades-{date_str}.zip"
            csv_name = f"{symbol}USDT-trades-{date_str}.csv"  # Assuming the CSV inside the zip has this name
            futures.append(executor.submit(download_and_unpack, url, file_name, csv_name))

if __name__ == "__main__":
    symbol = "BTC" 
    start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    end_date = datetime.now()  
    num_workers = 5  

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

    download_files(symbol, start_date, end_date, num_workers)
