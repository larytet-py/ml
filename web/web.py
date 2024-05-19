from flask import Flask, request, jsonify, render_template, send_from_directory
import clickhouse_connect
from datetime import datetime
import dateutil
import logging
import click

app = Flask(__name__, template_folder='templates', static_folder='static')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Define allowed symbols to prevent SQL injection via table names
TRADES_TABLES = {'BTC': 'trades_BTC', 'ETH': 'trades_ETH'}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/price_data')
def get_price_data():
    client = clickhouse_connect.get_client(host='localhost')

    symbol = request.args.get('symbol', default='BTC', type=str)
    if symbol not in TRADES_TABLES:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    start_iso = request.args.get('start', default='2024-02-01T00:00:00Z', type=str)
    end_iso = request.args.get('end', default='2024-02-01T01:00:00Z', type=str)

    # Parse datetime strings using dateutil.parser to handle both naive and aware formats
    start_date = dateutil.parser.parse(start_iso)
    end_date = dateutil.parser.parse(end_iso)

    # Convert datetime objects to strings that ClickHouse can handle (assuming UTC)
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    table_name = TRADES_TABLES[symbol]

    # Parameterized query for dates
    query = """
    SELECT 
        toFloat64(price) AS price,
        toUnixTimestamp64Milli(timestamp) AS time
    FROM 
        {}
    WHERE 
        timestamp BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY 
        timestamp ASC

    """.format(table_name)

    debug_query = query.replace("%(start_date)s", f"'{start_str}'").replace("%(end_date)s", f"'{end_str}'")
    logging.debug(debug_query)

    # Fetch the data as a DataFrame
    result = client.query_df(query, parameters={'start_date': start_str, 'end_date': end_str})
    
    # Convert DataFrame to a list of dictionaries (for JSON serialization)
    data = result.to_dict(orient='records')
    return jsonify(data)


@app.route('/ohlc_data')
def get_ohlc_data():
    client = clickhouse_connect.get_client(host='localhost')

    # Retrieve symbol from request arguments and validate
    symbol = request.args.get('symbol', default='BTC', type=str)
    if symbol not in TRADES_TABLES:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    # Retrieve and parse start and end date from request arguments
    start_iso = request.args.get('start', default='2024-02-01T00:00:00Z', type=str)
    end_iso = request.args.get('end', default='2024-02-01T01:00:00Z', type=str)
    start_date = dateutil.parser.parse(start_iso)
    end_date = dateutil.parser.parse(end_iso)

    # Retrieve interval duration from request arguments and validate
    interval_duration = request.args.get('interval', default=60, type=int)

    # Format dates for ClickHouse
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    # Get the table name from the symbol
    table_name = TRADES_TABLES[symbol]

    # Define the new query to fetch OHLC data
    query = f"""
    SELECT
        toUnixTimestamp64Milli(CAST(toStartOfInterval(timestamp, INTERVAL {interval_duration} SECOND) AS DateTime64)) AS time,
        toFloat64(any(price)) AS open,
        toFloat64(max(price)) AS high,
        toFloat64(min(price)) AS low,
        toFloat64(anyLast(price)) AS close
        -- count() AS num_trades
    FROM {table_name}
    WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY time
    ORDER BY time ASC
    """

    debug_query = query.replace("%(start_date)s", f"'{start_str}'").replace("%(end_date)s", f"'{end_str}'")
    logging.debug(debug_query)

    # Execute the query and fetch the result as a DataFrame
    result = client.query_df(query, parameters={'start_date': start_str, 'end_date': end_str})

    # Convert DataFrame to a list of dictionaries (for JSON serialization)
    data = result.to_dict(orient='records')
    return jsonify(data)

@app.route('/trades_density')
def get_trades_density():
    client = clickhouse_connect.get_client(host='localhost')

    # Retrieve symbol from request arguments and validate
    symbol = request.args.get('symbol', default='BTC', type=str)
    if symbol not in TRADES_TABLES:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    # Retrieve and parse start and end date from request arguments
    start_iso = request.args.get('start', default='2024-02-01T00:00:00Z', type=str)
    end_iso = request.args.get('end', default='2024-02-01T01:00:00Z', type=str)
    start_date = dateutil.parser.parse(start_iso)
    end_date = dateutil.parser.parse(end_iso)

    # Retrieve interval duration from request arguments and validate
    interval_duration = request.args.get('interval', default=60, type=int)

    # Format dates for ClickHouse
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    # Get the table name from the symbol
    table_name = TRADES_TABLES[symbol]

    # Define the new query to fetch OHLC data
    query = f"""
    SELECT
        toUnixTimestamp64Milli(CAST(toStartOfInterval(timestamp, INTERVAL {interval_duration} SECOND) AS DateTime64)) AS time,
        if(
            anyLast(price) = any(price),
            0,
            log(count() / (abs(toFloat64(anyLast(price)) - toFloat64(any(price))) / toFloat64(any(price))))
        ) AS price
    FROM {table_name}
    WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY time
    ORDER BY time ASC
    """

    debug_query = query.replace("%(start_date)s", f"'{start_str}'").replace("%(end_date)s", f"'{end_str}'")
    logging.debug(debug_query)

    # Execute the query and fetch the result as a DataFrame
    result = client.query_df(query, parameters={'start_date': start_str, 'end_date': end_str})

    # Convert DataFrame to a list of dictionaries (for JSON serialization)
    data = result.to_dict(orient='records')
    return jsonify(data)

@app.cli.command("runserver")
@click.option('--debug_level', default='INFO', help='Set the debug level')
def runserver(debug_level):
    numeric_level = getattr(logging, debug_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {debug_level}')
    logging.getLogger().setLevel(numeric_level)
    app.run(debug=True, port=8080)

if __name__ == '__main__':
    # Command line argument parsing for development purposes
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_level', default='INFO', help='Set the debug level')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.debug_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.debug_level}')
    logging.getLogger().setLevel(numeric_level)
    
    app.run(debug=True, port=8080)