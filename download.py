import argparse
import pandas as pd
import requests
import zipfile
import os
import sys
from datetime import datetime, timedelta
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import web.clickhouse_client as clickhouse_client_module
import re

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
    client = clickhouse_client_module.CLICKHOUSE_CLIENT.get_client(settings={'insert_deduplicate': '1'})

    column_names = ['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker', 'unknown_flag', 'timestamp']
    df = pd.read_csv(csv_name, header=None, names=column_names)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    if df.empty:
        logging.warning(f"No valid rows in {csv_name}. Skipping.")
        return
    df['time'] = df['time'].astype('int64')

    # Query to check the existence of all timestamps in the table
    first_timestamp, last_timestamp = df['time'].min(), df['time'].max()
    query = f"SELECT count(*) FROM {table_name} WHERE time BETWEEN {first_timestamp} AND {last_timestamp}"
    result = client.query(query)
    count_result = result.result_rows[0][0] if result.result_rows else 0
    if count_result >= 2:
        logging.info(f"Timestamps between {first_timestamp} and {last_timestamp} are present for {csv_name}. No update is needed.")
        return

    logging.info(f"One or both timestamps ({first_timestamp}, {last_timestamp}) are missing in {table_name} for {csv_name}. Inserting data.")

    # Binance changed trade time precision from milliseconds to microseconds.
    # Infer unit from numeric width to keep ingestion backward-compatible.
    time_unit = 'us' if len(str(int(first_timestamp))) >= 16 else 'ms'
    df['timestamp'] = pd.to_datetime(df['time'], unit=time_unit, utc=True).dt.tz_localize(None)

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
        for future in as_completed(futures):
            future.result()



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
        timestamp DateTime64(6) -- Microsecond precision
    )
    ENGINE = MergeTree
    PARTITION BY toDate(timestamp)
    ORDER BY (timestamp)
    SETTINGS index_granularity = 8192;
    """

    client = clickhouse_client_module.CLICKHOUSE_CLIENT.get_client()
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
    client = clickhouse_client_module.CLICKHOUSE_CLIENT.get_client()
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


def normalize_symbol(symbol):
    clean_symbol = re.sub(r'[^A-Za-z0-9_]', '', symbol.upper())
    if not clean_symbol:
        raise ValueError("Symbol must contain at least one alphanumeric character.")
    return clean_symbol


def fetch_twelvedata_1min(symbol, start_date, end_date, api_key):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1min",
        "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "UTC",
        "order": "ASC",
        "format": "JSON",
        "apikey": api_key,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "error":
        raise RuntimeError(f"TwelveData API error: {payload.get('message', 'unknown error')}")

    values = payload.get("values", [])
    if not values:
        logging.warning("TwelveData returned no rows for %s in requested range.", symbol)
        return pd.DataFrame()

    records = []
    for idx, value in enumerate(values):
        ts = pd.Timestamp(value["datetime"], tz="UTC")
        ts_naive = ts.tz_localize(None)
        epoch_ms = int(ts.timestamp() * 1000)
        close_price = float(value["close"])
        volume = float(value.get("volume", 0.0))
        records.append(
            {
                "id": idx,
                "price": close_price,
                "qty": 0.0,
                "base_qty": volume,
                "time": epoch_ms,
                "is_buyer_maker": False,
                "unknown_flag": True,
                "timestamp": ts_naive,
            }
        )

    return pd.DataFrame.from_records(records)


def insert_twelvedata_rows(table_name, df):
    if df.empty:
        return

    client = clickhouse_client_module.CLICKHOUSE_CLIENT.get_client(settings={"insert_deduplicate": "1"})
    min_time = df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S")
    max_time = df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")
    existing_query = f"""
    SELECT timestamp
    FROM {table_name}
    WHERE timestamp BETWEEN toDateTime('{min_time}') AND toDateTime('{max_time}')
    """
    existing_rows = client.query(existing_query).result_rows
    existing_timestamps = {row[0] for row in existing_rows}
    df_to_insert = df[~df["timestamp"].isin(existing_timestamps)]

    if df_to_insert.empty:
        logging.info("No new TwelveData rows to insert for %s.", table_name)
        client.close()
        return

    result = client.insert_df(table=table_name, df=df_to_insert)
    written = result.written_rows
    expected = len(df_to_insert)
    log_fn = logging.info if written == expected else logging.warning
    log_fn("Inserted %s/%s TwelveData rows into %s", written, expected, table_name)
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process trade data files.")
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbol to process, e.g., BTC')
    parser.add_argument('--download-twelvedata-1min', action='store_true', help='Download 1-minute OHLCV from TwelveData.')
    parser.add_argument('--twelvedata-symbol', type=str, default='GOOGL', help='TwelveData symbol to process (default: GOOGL).')
    parser.add_argument('--start_date', default=datetime.now() - timedelta(days=7), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', default=datetime.now(), type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help='End date in YYYY-MM-DD format')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes/threads')
    parser.add_argument('--disable-download', action='store_true', help='Disable the download functionality.')
    parser.add_argument('--clickhouse-username', type=str, default='default', help='Set Clickhouse user')
    parser.add_argument('--clickhouse-password', type=str, default='password', help='Set Clickhouse password')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    clickhouse_client_module.CLICKHOUSE_CLIENT = clickhouse_client_module.ClickhouseClient(
        username=args.clickhouse_username,
        password=args.clickhouse_password,
    )
    target_symbol = normalize_symbol(args.twelvedata_symbol if args.download_twelvedata_1min else args.symbol)
    table_name = f"trades_{target_symbol}"
    create_table_trades(table_name)
    if args.download_twelvedata_1min:
        api_key = os.environ.get("TWELVEDATA_API_KEY")
        if not api_key:
            raise EnvironmentError("TWELVEDATA_API_KEY env var is required for TwelveData download mode.")
        data = fetch_twelvedata_1min(
            symbol=args.twelvedata_symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            api_key=api_key,
        )
        insert_twelvedata_rows(table_name=table_name, df=data)
    elif not args.disable_download:
        download_files_process(
            symbol=target_symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            num_workers=args.num_workers,
            table_name=table_name,
        )
