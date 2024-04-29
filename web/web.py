from flask import Flask, jsonify
import clickhouse_connect

app = Flask(__name__)

# Connect to ClickHouse
client = clickhouse_connect.get_client(host='localhost')

@app.route('/price_data')
def get_price_data():
    # Query to select price and timestamp from your table
    query = """
    SELECT 
        toFloat64(price) AS price,  # Convert price to float for JSON compatibility
        formatDateTime(timestamp, '%Y-%m-%d %H:%M:%S') AS time  # Format timestamp to a more readable form
    FROM 
        trades_BTC
    ORDER BY 
        timestamp
    """
    # Fetch the data as a DataFrame
    result = client.query_df(query)
    
    # Convert DataFrame to a list of dictionaries (for JSON serialization)
    data = result.to_dict(orient='records')
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
