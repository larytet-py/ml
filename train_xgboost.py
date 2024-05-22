from clickhouse_connect import get_client
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Correct SQL Query
query = """
WITH 1*10*1000 as rows
SELECT 
    timestamp,
    price_stddev AS S30_stddev,
    roc AS S30_roc,
    dist_10_100,
    dist_100_1000,
    dist_1000_10000,
    dist_10000_100000,
    dist_100000_1000000,
    dist_1000000_10000000,
    dist_10000000_100000000,
    dist_greater_100000000
FROM ohlc_S30_BTC
ORDER BY timestamp
LIMIT rows
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
        feature_vector = feature_window[[
            'S30_stddev', 'S30_roc',
            'dist_10_100', 'dist_100_1000', 'dist_1000_10000',
            'dist_10000_100000', 'dist_100000_1000000', 'dist_1000000_10000000',
            'dist_10000000_100000000', 'dist_greater_100000000'
        ]].values.flatten()
        features.append(feature_vector)
        
        # Define the target
        target_value = target_window['S30_roc']
        targets.append(target_value)

# Debugging: Check lengths of features and targets
print(f'Number of feature sets: {len(features)}')
print(f'Number of target values: {len(targets)}')

# Convert to DataFrame and Series
if features:  # Check if features list is not empty
    feature_columns = [
        'S30_stddev1', 'S30_roc1',
        'dist_10_1001', 'dist_100_10001', 'dist_1000_100001',
        'dist_10000_1000001', 'dist_100000_10000001', 'dist_1000000_100000001',
        'dist_10000000_1000000001', 'dist_greater_1000000001',
        'S30_stddev2', 'S30_roc2',
        'dist_10_1002', 'dist_100_10002', 'dist_1000_100002',
        'dist_10000_1000002', 'dist_100000_10000002', 'dist_1000000_100000002',
        'dist_10000000_1000000002', 'dist_greater_1000000002',
        'S30_stddev3', 'S30_roc3',
        'dist_10_1003', 'dist_100_10003', 'dist_1000_100003',
        'dist_10000_1000003', 'dist_100000_10000003', 'dist_1000000_100000003',
        'dist_10000000_1000000003', 'dist_greater_1000000003',
        'S30_stddev4', 'S30_roc4',
        'dist_10_1004', 'dist_100_10004', 'dist_1000_100004',
        'dist_10000_1000004', 'dist_100000_10000004', 'dist_1000000_100000004',
        'dist_10000000_1000000004', 'dist_greater_1000000004'
    ]
    
    X = pd.DataFrame(features, columns=feature_columns)
    y = pd.Series(targets)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    if len(features) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_estimators': [5, 50, 100, 200],
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'max_depth': [1, 3, 5, 7],
            'subsample': [0.2, 0.6, 0.8, 1.0],
            'colsample_bytree': [0.2, 0.6, 0.8, 1.0]
        }

        model = xgb.XGBRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.4f}')
        print(f'R^2 Score: {r2:.4f}')

        # Debugging: Print some of the predictions and actual values
        print("Predictions vs Actual:")
        for pred, actual in zip(y_pred[:10], y_test[:10]):
            print(f'Predicted: {pred:.4f}, Actual: {actual:.4f}')

        print(f'Best hyperparameters: {grid_search.best_params_}')
    else:
        print("Not enough data for train-test split. Ensure you have sufficient data.")
else:
    print("No features generated. Check the rolling window logic.")



