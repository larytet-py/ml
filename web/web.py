import argparse

import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
import clickhouse_connect
import dateutil
import logging

app = Flask(__name__, template_folder='templates', static_folder='static')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Define allowed symbols to prevent SQL injection via table names
TRADES_TABLES = {'BTC': 'trades_BTC', 'ETH': 'trades_ETH'}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

def get_client():
    return clickhouse_connect.get_client(host='localhost')

def validate_and_parse_args():
    symbol = request.args.get('symbol', default='BTC', type=str)
    if symbol not in TRADES_TABLES:
        return None, jsonify({'error': 'Invalid symbol'}), 400

    start_iso = request.args.get('start', default='2024-02-01T00:00:00Z', type=str)
    end_iso = request.args.get('end', default='2024-02-01T01:00:00Z', type=str)

    start_date = dateutil.parser.parse(start_iso)
    end_date = dateutil.parser.parse(end_iso)

    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    interval_duration = request.args.get('interval', default=60, type=int)

    table_name = TRADES_TABLES[symbol]

    return { 
        'start_date': start_str,
        'end_date': end_str,
        'table_name': table_name,
        'interval_duration': interval_duration
    }, None, 200


def execute_query(query, parameters):
    def format_param(value):
        if isinstance(value, str):
            return f"'{value}'"
        return str(value)

    param_dict = {
        'start_date': format_param(parameters['start_date']),
        'end_date': format_param(parameters['end_date']),
        'table_name': parameters['table_name'],  # Assuming table_name doesn't need quotes
        'interval_duration': format_param(parameters['interval_duration'])
    }    
    if 'period' in parameters:
        param_dict['period'] = format_param(parameters['period'])
    
    debug_query = query % param_dict
    logging.debug(debug_query)
      
    # Execute the query and fetch the result as a DataFrame
    result = get_client().query_df(query, parameters)

    # Convert DataFrame to a list of tuples
    data = list(result.itertuples(index=False, name=None))
    return jsonify(data)

@app.route('/price_data')
def get_price_data():
    parameters, error_response, status = validate_and_parse_args()
    if error_response:
        return error_response, status

    query = """
    SELECT 
        toUnixTimestamp64Milli(timestamp) AS time,
        toFloat64(price) AS price
    FROM 
        %(table_name)s
    WHERE 
        timestamp BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY 
        timestamp ASC
    """

    data = execute_query(query, parameters)
    return data

@app.route('/price_ma')
def get_price_ma():
    parameters, error_response, status = validate_and_parse_args()
    if error_response:
        return error_response, status

    parameters['period'] = request.args.get('period', default=60, type=int)

    query = """
    SELECT
        -- Convert timestamp to milliseconds and group by intervals of specified duration
        toUnixTimestamp64Milli(
            CAST(toStartOfInterval(timestamp, INTERVAL %(interval_duration)s SECOND) AS DateTime64)
        ) AS time,      
        -- Calculate the moving average of the price using window function
        avg(price) OVER (
            -- Order data by timestamp in milliseconds grouped by intervals of specified duration
            ORDER BY toUnixTimestamp64Milli(
                CAST(toStartOfInterval(timestamp, INTERVAL %(interval_duration)s SECOND) AS DateTime64)
            )
            -- Define window frame: from 'period' number of preceding rows to the current row
            ROWS BETWEEN %(period)s PRECEDING AND CURRENT ROW
        ) AS moving_average
    FROM 
        %(table_name)s
    WHERE 
        timestamp BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY 
        time ASC
    """

    data = execute_query(query, parameters)
    return data

@app.route('/ohlc_data')
def get_ohlc_data():
    parameters, error_response, status = validate_and_parse_args()
    if error_response:
        return error_response, status

    query = """
    SELECT
        toUnixTimestamp64Milli(CAST(toStartOfInterval(timestamp, INTERVAL %(interval_duration)s SECOND) AS DateTime64)) AS time,
        toFloat64(any(price)) AS open,
        toFloat64(max(price)) AS high,
        toFloat64(min(price)) AS low,
        toFloat64(anyLast(price)) AS close
        -- count() AS num_trades
    FROM %(table_name)s
    WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY time
    ORDER BY time ASC
    """

    data = execute_query(query, parameters)
    return data

@app.route('/trades_density')
def get_trades_density():
    parameters, error_response, status = validate_and_parse_args()
    if error_response:
        return error_response, status

    query = """
    WITH
        trades_1s AS (
            SELECT
                toUnixTimestamp64Milli(CAST(toStartOfInterval(timestamp, INTERVAL 1 SECOND) AS DateTime64)) AS time,
                toFloat64(any(price)) AS open,
                toFloat64(anyLast(price)) AS close,
                count() AS num_trades
            FROM %(table_name)s
            WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY time
            ORDER BY time ASC
        ),
        rolling_metrics AS (
            SELECT
                time,
                sum(num_trades) OVER (
                    ORDER BY time
                    ROWS BETWEEN %(interval_duration)s PRECEDING AND CURRENT ROW
                ) AS rolling_num_trades,
                abs(
                    (max(close) OVER (
                        ORDER BY time
                        ROWS BETWEEN %(interval_duration)s PRECEDING AND CURRENT ROW
                    ) - min(open) OVER (
                        ORDER BY time
                        ROWS BETWEEN %(interval_duration)s PRECEDING AND CURRENT ROW
                    )) / nullif(min(open) OVER (
                        ORDER BY time
                        ROWS BETWEEN %(interval_duration)s PRECEDING AND CURRENT ROW
                    ), 0)
                ) AS rolling_roc
            FROM trades_1s
        )
    SELECT
        time,
        if(
            rolling_roc = 0,
            last_value(log(rolling_num_trades / nullif(rolling_roc, 0))) OVER (
                ORDER BY time
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            log(
                rolling_num_trades / nullif(rolling_roc, 0)
            )
        ) AS forward_filled_density
    FROM rolling_metrics
    ORDER BY time ASC
    """

    data = execute_query(query, parameters)
    return data

@app.route('/autocorrelation')
def get_autocorrelation():
    parameters, error_response, status = validate_and_parse_args()
    if error_response:
        return error_response, status

    window_size = request.args.get('window_size', default=30, type=int)

    query = """
    SELECT timestamp, close_price
    FROM %(table_name)s
    WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY timestamp ASC
    """
    
    data = execute_query(query, parameters)
    df = pd.DataFrame(data, columns=['timestamp', 'close_price'])
    
    autocorrelations = []

    for i in range(window_size, len(df) + 1):
        window = df['close_price'][i-window_size:i]
        mean_window = np.mean(window)
        autocorrelation = np.sum((window[:-1] - mean_window) * (window[1:] - mean_window)) / np.sum((window - mean_window) ** 2)
        autocorrelations.append((df['timestamp'][i-1], autocorrelation))

    return jsonify(autocorrelations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_level', default='INFO', help='Set the debug level')
    parser.add_argument('--port', default=8080, help='HTTP port to bind')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.debug_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.debug_level}')
    logging.getLogger().setLevel(numeric_level)
    
    app.run(debug=True, port=args.port)