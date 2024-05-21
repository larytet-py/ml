from clickhouse_connect import get_client
import pandas as pd


query = """
SELECT 
    s5.timestamp AS timestamp,
    s30.price_stddev AS S30_stddev,
    s30.roc AS S30_roc,
    s5.price_stddev AS S5_stddev,
    s5.roc AS S5_roc
FROM 
    (SELECT 
        timestamp,
        price_stddev,
        roc
     FROM ohlc_S5_BTC) AS s5
JOIN 
    (SELECT 
        timestamp,
        price_stddev,
        roc
     FROM ohlc_S30_BTC) AS s30
ON 
    s30.timestamp = toStartOfMinute(s5.timestamp) + INTERVAL (toMinute(s5.timestamp) % 30) SECOND
ORDER BY s5.timestamp
"""

client = get_client()
result_df = client.query_df(query)

result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
result_df.set_index('timestamp', inplace=True)

# Create rolling windows of 5 minutes
rolling_windows = result_df.rolling(window='5T', closed='right')

# Extract features for the first 4 minutes and the target for the 5th minute
features = []
targets = []

for window in rolling_windows:
    if len(window) == 5:
        feature_window = window.iloc[:-1]  # First 4 minutes
        target_window = window.iloc[-1]    # 5th minute
        
        # Extract the features
        feature_vector = feature_window[['S30_stddev', 'S30_roc', 'S5_stddev', 'S5_roc']].values.flatten()
        features.append(feature_vector)
        
        # Define the target
        target_value = target_window['S5_roc']
        targets.append(target_value)

X = pd.DataFrame(features, columns=[
    'S30_stddev1', 'S30_roc1', 'S5_stddev1', 'S5_roc1',
    'S30_stddev2', 'S30_roc2', 'S5_stddev2', 'S5_roc2',
    'S30_stddev3', 'S30_roc3', 'S5_stddev3', 'S5_roc3',
    'S30_stddev4', 'S30_roc4', 'S5_stddev4', 'S5_roc4'
])
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
