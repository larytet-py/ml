from flask import Flask, request, jsonify
import clickhouse_connect

app = Flask(__name__)

# Connect to ClickHouse
client = clickhouse_connect.get_client(host='localhost')

@app.route('/price_data')
def get_price_data():
    symbol = request.args.get('symbol', default='BTC', type=str)  # default to BTC if not specified
    start_date = request.args.get('start', default='2024-01-01T00:00:00', type=str)
    end_date = request.args.get('end', default='2024-01-01T04:00:00', type=str)
    query = f"""
    SELECT 
        toFloat64(price) AS price,
        formatDateTime(timestamp, '%Y-%m-%d %H:%M:%S') AS time
    FROM 
        trades_{symbol}
    WHERE 
        timestamp BETWEEN '{{start_date}}' AND '{{end_date}}'
    ORDER BY 
        timestamp
    """

    # Fetch the data as a DataFrame
    result = client.query_df(query, params={'start_date': start_date, 'end_date': end_date})
    
    # Convert DataFrame to a list of dictionaries (for JSON serialization)
    data = result.to_dict(orient='records')
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
