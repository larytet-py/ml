from dataclasses import asdict, dataclass
from flask import Flask, request, jsonify, render_template, send_from_directory
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

@dataclass
class QueryParameters:
    start_date: str
    end_date: str
    table_name: str
    interval_duration: int

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

    return QueryParameters(start_str, end_str, table_name, interval_duration), None, 200

def execute_query(query, parameters):
    parameters = asdict(parameters)

    debug_query = query % {
            'start_date': parameters['start_date'],
            'end_date': parameters['end_date'],
            'table_name': parameters['table_name'],
            'interval_duration': parameters['interval_duration']
        }   
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
    SELECT
        toUnixTimestamp64Milli(CAST(toStartOfInterval(timestamp, INTERVAL %(interval_duration)s SECOND) AS DateTime64)) AS time,
        if(
            anyLast(price) = any(price),
            0,
            log(count() / (abs(toFloat64(anyLast(price)) - toFloat64(any(price))) / toFloat64(any(price))))
        ) AS density
    FROM %(table_name)s
    WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY time
    ORDER BY time ASC
    """

    data = execute_query(query, parameters)
    return data

if __name__ == '__main__':
    # Command line argument parsing for development purposes
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_level', default='INFO', help='Set the debug level')
    parser.add_argument('--port', default=8080, help='HTTP port to bind')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.debug_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.debug_level}')
    logging.getLogger().setLevel(numeric_level)
    
    app.run(debug=True, port=args.port)