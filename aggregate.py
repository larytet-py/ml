from clickhouse_connect import get_client
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

def create_table_ohlc(table_name):
    check_table_exists_query = f"EXISTS TABLE {table_name}"
    create_table_query = f"""
    CREATE TABLE {table_name}
    (
        time_start DateTime,
        time_end DateTime,
        open Decimal(18, 8),
        high Decimal(18, 8),
        low Decimal(18, 8),
        close Decimal(18, 8),
        volume Decimal(12, 8)
    )
    ENGINE = MergeTree
    ORDER BY (time_start)
    SETTINGS index_granularity = 8192;    
    """

    client = get_client()
    result = client.query(check_table_exists_query)
    count_result = result.result_rows[0][0] if result.result_rows else 0
    if count_result > 0:
        logging.debug(f"Table '{table_name}' already exists.")
        return
    
    client.query(create_table_query)
    logging.info(f"Table '{table_name}' created successfully.")

def fetch_and_aggregate_data():
    client = get_client()
    trades_query = "SELECT time, price, base_qty FROM trades_BTC"
    df = client.query_dataframe(trades_query)
    
    df['time'] = pd.to_datetime(df['time'], unit='ms')  # Assuming 'time' is in milliseconds
    df.set_index('time', inplace=True)

    # Resample to 1-second intervals
    ohlc = df['price'].resample('1S').ohlc()
    volume = df['base_qty'].resample('1S').sum()

    # Prepare DataFrame for insertion
    ohlc['volume'] = volume
    ohlc.reset_index(inplace=True)
    ohlc.columns = ['time_start', 'open', 'high', 'low', 'close', 'volume']
    ohlc['time_end'] = ohlc['time_start'] + timedelta(seconds=1)

    # Handling missing data by filling with last known values
    ohlc.fillna(method='ffill', inplace=True)
    
    return ohlc

def insert_data(df, table_name):
    client = get_client()
    # Convert DataFrame to ClickHouse-insertable format or use a direct insertion method if available in your library
    # Example (make sure to replace with actual implementation):
    client.insert_dataframe(table_name, df)
    logging.info("Data inserted successfully.")

def fetch_trades_and_insert():
    create_table_ohlc("bars_S1")
    df = fetch_and_aggregate_data()
    insert_data(df, "bars_S1")

if __name__ == '__main__':
    fetch_trades_and_insert()
