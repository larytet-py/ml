from clickhouse_connect import get_client
import pandas as pd


query = """
SELECT 
    toStartOfFiveMinute(timestamp) AS interval_start,
    avg(roc) AS avg_roc,
    avg(price_stddev) AS avg_stddev
FROM ohlc_S5_BTC
GROUP BY interval_start
UNION ALL
SELECT 
    toStartOfFiveMinute(timestamp) AS interval_start,
    avg(roc) AS avg_roc,
    avg(price_stddev) AS avg_stddev
FROM ohlc_S30_BTC
GROUP BY interval_start
ORDER BY interval_start
"""

client = get_client()

# Function to fetch data in chunks
def fetch_data_in_chunks(client, query, batch_size=10000):
    offset = 0
    while True:
        batch_query = f"""
        {query}
        LIMIT {batch_size} OFFSET {offset}
        """
        df = client.query_df(batch_query)
        if df.empty:
            break
        yield df
        offset += batch_size

features = []
targets = []

for batch_df in fetch_data_in_chunks(client, query):
    batch_df['interval_start'] = pd.to_datetime(batch_df['interval_start'])
    batch_df.set_index('interval_start', inplace=True)
    
    rolling_windows = batch_df.rolling(window='5T', closed='right')
    
    for window in rolling_windows:
        if len(window) == 5:
            feature_window = window.iloc[:-1]  # First 4 minutes
            target_window = window.iloc[-1]    # 5th minute
            
            # Extract the features
            feature_vector = feature_window[['avg_roc', 'avg_stddev']].values.flatten()
            features.append(feature_vector)
            
            # Define the target
            target_value = target_window['avg_roc']
            targets.append(target_value)

X = pd.DataFrame(features, columns=['roc1', 'stddev1', 'roc2', 'stddev2', 'roc3', 'stddev3', 'roc4', 'stddev4'])
y = pd.Series(targets)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
