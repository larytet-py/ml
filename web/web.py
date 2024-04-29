from flask import Flask, request, jsonify, render_template
import clickhouse_connect
from datetime import datetime
import dateutil 

app = Flask(__name__, template_folder='templates')

# Connect to ClickHouse
client = clickhouse_connect.get_client(host='localhost')

# Define allowed symbols to prevent SQL injection via table names
SYMBOLS = {'BTC': 'trades_BTC', 'ETH': 'trades_ETH'}

@app.route('/')
def index():
    # Render your HTML file
    return render_template('index.html')

@app.route('/price_data')
def get_price_data():
    symbol = request.args.get('symbol', default='BTC', type=str)
    if symbol not in SYMBOLS:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    start_iso = request.args.get('start', default='2024-01-01T00:00:00Z', type=str)
    end_iso = request.args.get('end', default='2024-01-01T01:00:00Z', type=str)

    # Parse datetime strings using dateutil.parser to handle both naive and aware formats
    start_date = dateutil.parser.parse(start_iso)
    end_date = dateutil.parser.parse(end_iso)

    # Convert datetime objects to strings that ClickHouse can handle (assuming UTC)
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    table_name = SYMBOLS[symbol]

    # Parameterized query for dates
    query = """
    SELECT 
        toFloat64(price) AS price,
        formatDateTime(timestamp, '%%Y-%%m-%%d %%H:%%M:%%S') AS time
    FROM 
        {}
    WHERE 
        timestamp BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY 
        timestamp ASC
    """.format(table_name)

    # Fetch the data as a DataFrame
    result = client.query_df(query, parameters={'start_date': start_str, 'end_date': end_str})
    
    # Convert DataFrame to a list of dictionaries (for JSON serialization)
    data = result.to_dict(orient='records')
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
