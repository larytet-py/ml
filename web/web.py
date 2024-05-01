from flask import Flask, request, jsonify, render_template, send_from_directory
import clickhouse_connect
from datetime import datetime
import dateutil 

app = Flask(__name__, template_folder='templates')


# Define allowed symbols to prevent SQL injection via table names
TRADES_TABLES = {'BTC': 'trades_BTC', 'ETH': 'trades_ETH'}
OHLC_TABLES = {'BTC': 'ohlc_S1_BTC', 'ETH': 'ohlc_S1_ETH'}

@app.route('/')
def index():
    # Render your HTML file
    return render_template('index.html')

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

    # Fetch the data as a DataFrame
    result = client.query_df(query, parameters={'start_date': start_str, 'end_date': end_str})
    print(result)
    
    # Convert DataFrame to a list of dictionaries (for JSON serialization)
    data = result.to_dict(orient='records')
    return jsonify(data)

@app.route('/ohlc_data')
def get_ohlc_data():
    client = clickhouse_connect.get_client(host='localhost')

    # Retrieve symbol from request arguments and validate
    symbol = request.args.get('symbol', default='BTC', type=str)
    if symbol not in OHLC_TABLES:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    # Retrieve and parse start and end date from request arguments
    start_iso = request.args.get('start', default='2024-02-01T00:00:00Z', type=str)
    end_iso = request.args.get('end', default='2024-02-01T01:00:00Z', type=str)
    start_date = dateutil.parser.parse(start_iso)
    end_date = dateutil.parser.parse(end_iso)

    # Format dates for ClickHouse
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    # Get the table name from the symbol
    table_name = TRADES_TABLES[symbol]

    # Define the new query to fetch price data and treat it as OHLC
    query = f"""
    SELECT 
        toFloat64(price) AS price,
        toUnixTimestamp64Milli(timestamp) AS time
    FROM 
        {table_name}
    WHERE 
        timestamp BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY 
        timestamp ASC
    """

    # Execute the query and fetch the result as a DataFrame
    result = client.query_df(query, parameters={'start_date': start_str, 'end_date': end_str})

    # Process the DataFrame to set OHLC values to the same price
    data = result.to_dict(orient='records')
    ohlc_data = [
        {'open': item['price'], 'high': item['price'], 'low': item['price'], 'close': item['price'], 'time': item['time']}
        for item in data
    ]

    # Return JSON response
    return jsonify(ohlc_data)

@app.route('/panels.json')
def panels_json():
    return send_from_directory('static', 'panels.json')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
