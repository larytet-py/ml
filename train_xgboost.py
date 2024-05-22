from clickhouse_connect import get_client
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Correct SQL Query
query = """
WITH 1000*1000 AS rows, 
     bin_intervals AS (SELECT 
        CASE
            WHEN log10(base_qty) < 1 THEN '<10^1'
            WHEN log10(base_qty) >= 1 AND log10(base_qty) < 2 THEN '10^1 - 10^2'
            WHEN log10(base_qty) >= 2 AND log10(base_qty) < 3 THEN '10^2 - 10^3'
            WHEN log10(base_qty) >= 3 AND log10(base_qty) < 4 THEN '10^3 - 10^4'
            WHEN log10(base_qty) >= 4 AND log10(base_qty) < 5 THEN '10^4 - 10^5'
            WHEN log10(base_qty) >= 5 AND log10(base_qty) < 6 THEN '10^5 - 10^6'
            WHEN log10(base_qty) >= 6 AND log10(base_qty) < 7 THEN '10^6 - 10^7'
            WHEN log10(base_qty) >= 7 AND log10(base_qty) < 8 THEN '10^7 - 10^8'
            ELSE '>10^8'
        END AS base_qty_bin,
        timestamp
     FROM (
        SELECT toFloat64(base_qty) AS base_qty, timestamp
        FROM trades_BTC
     )
),
bin_counts AS (
    SELECT
        timestamp,
        base_qty_bin,
        count(*) AS bin_count
    FROM bin_intervals
    GROUP BY timestamp, base_qty_bin
),
total_counts AS (
    SELECT
        timestamp,
        sum(bin_count) AS total_count
    FROM bin_counts
    GROUP BY timestamp
)
SELECT 
    s5.timestamp AS timestamp,
    s30.price_stddev AS S30_stddev,
    s30.roc AS S30_roc,
    s5.price_stddev AS S5_stddev,
    s5.roc AS S5_roc,
    sum(if(base_qty_bin='<10^1', bin_count, 0)) / max(total_count) AS ratio_10_1,
    sum(if(base_qty_bin='10^1 - 10^2', bin_count, 0)) / max(total_count) AS ratio_10_1_10_2,
    sum(if(base_qty_bin='10^2 - 10^3', bin_count, 0)) / max(total_count) AS ratio_10_2_10_3,
    sum(if(base_qty_bin='10^3 - 10^4', bin_count, 0)) / max(total_count) AS ratio_10_3_10_4,
    sum(if(base_qty_bin='10^4 - 10^5', bin_count, 0)) / max(total_count) AS ratio_10_4_10_5,
    sum(if(base_qty_bin='10^5 - 10^6', bin_count, 0)) / max(total_count) AS ratio_10_5_10_6,
    sum(if(base_qty_bin='10^6 - 10^7', bin_count, 0)) / max(total_count) AS ratio_10_6_10_7,
    sum(if(base_qty_bin='10^7 - 10^8', bin_count, 0)) / max(total_count) AS ratio_10_7_10_8,
    sum(if(base_qty_bin='>10^8', bin_count, 0)) / max(total_count) AS ratio_greater_10_8
FROM 
    (SELECT 
        timestamp,
        price_stddev,
        roc
     FROM ohlc_S5_BTC LIMIT rows) AS s5
JOIN 
    (SELECT 
        timestamp,
        price_stddev,
        roc
     FROM ohlc_S30_BTC LIMIT rows) AS s30
ON 
    s30.timestamp = toStartOfMinute(s5.timestamp) + INTERVAL (toSecond(s5.timestamp) % 30) SECOND
LEFT JOIN bin_counts ON s5.timestamp = bin_counts.timestamp
LEFT JOIN total_counts ON s5.timestamp = total_counts.timestamp
GROUP BY s5.timestamp, s30.price_stddev, s30.roc, s5.price_stddev, s5.roc
ORDER BY s5.timestamp
"""

# Fetch Data
client = get_client()
result_df = client.query_df(query)

# Debugging: Print the first few rows of result_df
print("First few rows of result_df:")
print(result_df.head())

# Check for NULLs and handle them if necessary
result_df.dropna(inplace=True)

# Convert timestamp to datetime
result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
result_df.set_index('timestamp', inplace=True)

# Debugging: Print the length of result_df after dropping NA
print(f"Number of rows in result_df after dropping NA: {len(result_df)}")

# Extract features for the first 4 minutes and the target for the 5th minute
features = []
targets = []

# Use a list of indices for the rolling window
window_indices = range(len(result_df) - 5 + 1)

for i in window_indices:
    window = result_df.iloc[i:i + 5]
    if len(window) == 5:
        feature_window = window.iloc[:-1]  # First 4 minutes
        target_window = window.iloc[-1]    # 5th minute
        
        # Extract the features
        feature_vector = feature_window[['S30_stddev', 'S30_roc', 'S5_stddev', 'S5_roc']].values.flatten()
        features.append(feature_vector)
        
        # Define the target
        target_value = target_window['S5_roc']
        targets.append(target_value)

# Debugging: Check lengths of features and targets
print(f'Number of feature sets: {len(features)}')
print(f'Number of target values: {len(targets)}')

# Convert to DataFrame and Series
if features:  # Check if features list is not empty
    X = pd.DataFrame(features, columns=[
        'S30_stddev1', 'S30_roc1', 'S5_stddev1', 'S5_roc1',
        'S30_stddev2', 'S30_roc2', 'S5_stddev2', 'S5_roc2',
        'S30_stddev3', 'S30_roc3', 'S5_stddev3', 'S5_roc3',
        'S30_stddev4', 'S30_roc4', 'S5_stddev4', 'S5_roc4'
    ])
    y = pd.Series(targets)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    import matplotlib.pyplot as plt
    import seaborn as sns

    # Inspect feature distributions
    for column in X.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(X[column], bins=50, kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

    # Inspect target variable distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y, bins=50, kde=True)
    plt.title('Distribution of S5_roc (Target Variable)')
    plt.show()

    # Split the data into training and testing sets
    if len(features) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Create and train the model with modified parameters
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.4f}')
        print(f'R^2 Score: {r2:.4f}')

        # Debugging: Print some of the predictions and actual values
        print("Predictions vs Actual:")
        for pred, actual in zip(y_pred[:10], y_test[:10]):
            print(f'Predicted: {pred:.4f}, Actual: {actual:.4f}')
    else:
        print("Not enough data for train-test split. Ensure you have sufficient data.")
else:
    print("No features generated. Check the rolling window logic.")