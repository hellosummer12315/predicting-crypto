# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import linregress
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, \
    recall_score, f1_score, roc_curve
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
import backtrader as bt
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import sys
from io import StringIO
from sklearn.inspection import permutation_importance

### Initial Setup and Imports
# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

warnings.filterwarnings('ignore')

# Create visualization folder
if not os.path.exists('vis'):
    os.makedirs('vis')

# Set font for plots (using Arial for English support)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Log redirection class
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# Initialize log file
log_file = open('log.txt', 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

# Load prices data
print("Loading prices data...")
url = 'https://drive.google.com/uc?id=1P_5ykYLd5521QUdCxC_cMytdJ3PqESTw'
prices = pd.read_csv(url, parse_dates=True, index_col=0)
# Remove duplicate indices, keeping the first occurrence to prevent reindexing errors
print(f"Original prices shape: {prices.shape}")
prices = prices[~prices.index.duplicated(keep='first')]
print(f"After removing duplicates, prices shape: {prices.shape}")
print("✓ Loaded prices")

### Feature Engineering
# Compute a variety of technical indicators and add them to `prices`.
print("Starting feature engineering...")

# MACD (12-EMA minus 26-EMA) + 9-day signal line
prices['MACD'] = prices['close'].ewm(span=12, adjust=False).mean() - prices['close'].ewm(span=26, adjust=False).mean()
prices['MACD_signal'] = prices['MACD'].ewm(span=9, adjust=False).mean()

# Stochastic %K over 14 days
window_k = 14
low_k = prices['low'].rolling(window_k).min()
high_k = prices['high'].rolling(window_k).max()
prices['Stoch_%K'] = 100 * (prices['close'] - low_k) / (high_k - low_k)

# Donchian channel (20-day high/low)
window_d = 20
prices['Donchian_upper'] = prices['high'].rolling(window_d).max()
prices['Donchian_lower'] = prices['low'].rolling(window_d).min()

# Bollinger Band width (20-day SMA ±2σ)
window_b = 20
ma20 = prices['close'].rolling(window_b).mean()
std20 = prices['close'].rolling(window_b).std()
prices['BB_width'] = ((ma20 + 2 * std20) - (ma20 - 2 * std20)) / ma20 * 100

# Log returns + 7-day raw return
prices['LogReturn'] = np.log(prices['close'] / prices['close'].shift(1))
prices['Return_7d'] = prices['close'].pct_change(7)

# Rolling skewness & kurtosis over 30 days of daily returns
window_s = 30
prices['Return_1d'] = prices['close'].pct_change(1)
prices['Skew30'] = prices['Return_1d'].rolling(window_s).skew()
prices['Kurt30'] = prices['Return_1d'].rolling(window_s).kurt()

# Lagged return (yesterday’s log-return)
prices['PrevReturn1'] = prices['LogReturn'].shift(1)

# Regime feature via a 3-state Gaussian HMM on (LogReturn, BB_width)
hmm_feats = ['LogReturn', 'BB_width']
hmm_data = prices[hmm_feats].dropna()
hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
hmm.fit(hmm_data)
prices.loc[hmm_data.index, 'regime'] = hmm.predict(hmm_data)

# Consolidate feature names
feature_cols = [
    'MACD', 'MACD_signal', 'Stoch_%K', 'Donchian_upper', 'Donchian_lower',
    'BB_width', 'LogReturn', 'Return_7d', 'Skew30', 'Kurt30', 'PrevReturn1', 'regime'
]

# Show first few rows of price + features
print("✓ Feature engineering completed. Sample features:")
print(prices[['close'] + feature_cols].dropna().head(5))


### Visualization
print("Generating feature visualizations...")

# Plot 1: Closing Price vs. MACD Indicator ---
# Plot close price and MACD series
fig_macd, ax_price = plt.subplots(figsize=(10, 6))
ax_macd = ax_price.twinx()
ax_price.plot(prices.index, prices['close'], label='Close Price', color='blue')
ax_macd.plot(prices.index, prices['MACD'], label='MACD', color='orange')
ax_macd.plot(prices.index, prices['MACD_signal'], label='MACD Signal Line', color='green', linestyle='--')
# Titles and axis labels
ax_price.set_title('Close Price and MACD Indicator Over Time')
ax_price.set_xlabel('Date')
ax_price.set_ylabel('Close Price', color='blue')
ax_macd.set_ylabel('MACD', color='orange')
# Zero line for reference
ax_macd.axhline(0, color='gray', linestyle=':')
# Combine legends from both axes
h_price, l_price = ax_price.get_legend_handles_labels()
h_macd, l_macd = ax_macd.get_legend_handles_labels()
ax_price.legend(h_price + h_macd, l_price + l_macd, loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('vis/MACD_vs_Price.png')
plt.close()
print("✓ MACD chart saved to vis/MACD_vs_Price.png")

# Plot 2: Closing Price and Bollinger Band Width ---
# Top: close price
# Bottom: BB width
fig_bb, (ax_price_bb, ax_bw) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_price_bb.plot(prices.index, prices['close'], label='Close Price', color='blue')
ax_bw.plot(prices.index, prices['BB_width'], label='Bollinger Band Width (%)', color='purple')
ax_price_bb.set_ylabel('Close Price')
ax_bw.set_xlabel('Date')
ax_bw.set_ylabel('Bollinger Band Width (%)', color='purple')
ax_price_bb.set_title('Close Price and Bollinger Band Width')
ax_bw.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('vis/BB_Width_vs_Price.png')
plt.close()
print("✓ Bollinger Band Width chart saved to vis/BB_Width_vs_Price.png")





### Label Creation
print("Starting trend scanning label creation...")

# Compute the t-value
def t_value(segment: np.ndarray) -> float:
    # Design matrix with intercept and time index
    n = segment.size
    X = np.column_stack((np.ones(n), np.arange(n)))
    # Fit ordinary least squares
    fit = sm.OLS(segment, X).fit()
    # Return the t-value for the slope coefficient
    return float(fit.tvalues[1])

# Label each timestamp with its most significant trend
def get_trend_labels(series: pd.Series, span: tuple = (5, 20), forward: bool = True) -> pd.DataFrame:
    # Observation span is the minimum and maximum window lengths
    min_w, max_w = span
    # DataFrame to hold results
    out = pd.DataFrame(index=series.index, columns=['t1', 'tVal', 'bin', 'windowSize'])
    N = len(series)
    # Loop through every timestamp
    for ts in series.index:
        # Find the integer position of current timestamp
        i = series.index.get_loc(ts)
        # Skip if not enough data forward or backward
        if forward and i + max_w >= N: continue
        if not forward and i - max_w < 0: continue
        # Dictionary to hold t-values for each window
        stats = {}
        # Loop through every window length
        for w in range(min_w, max_w + 1):
            # Start and end indices based on look direction
            start = i if forward else i - w
            end = i + w if forward else i
            # Extract the price segment
            seg = series.values[start:end + 1]
            # Need at least two points to fit a line
            if len(seg) < 2: continue
            # Compute and store the slope t-value
            stats[w] = t_value(seg)
        # Skip if no valid windows found
        if not stats: continue
        # Pick the window with highest absolute t-value
        best_w = max(stats, key=lambda k: abs(stats[k]))
        # Compute the index of the window's endpoint
        end_idx = i + best_w if forward else i - best_w
        # Record the end timestamp, t-value, direction, and window size
        out.loc[ts] = (series.index[end_idx], stats[best_w], int(np.sign(stats[best_w])), best_w)
    # Format labels DataFrame
    out = out.dropna(subset=['bin'])
    out['bin'] = out['bin'].astype(int)
    out['windowSize'] = out['windowSize'].astype(int)
    out['tVal'] = out['tVal'].clip(-20, 20)
    out['t1'] = pd.to_datetime(out['t1'])
    return out

# Apply to full close-price series
close_prices = prices['close']
label_ful = get_trend_labels(close_prices, span=(5, 20), forward=True)

# Print first few rows of the trend-scanning label DataFrame
print("✓ Trend scanning labels created. Sample labels:")
print(label_ful.head())

# Get the matching close prices at those event timestamps
# Plot the trending scanning labels
plt.figure(figsize=(10, 5))
sc = plt.scatter(x=label_ful['t1'], y=close_prices.loc[label_ful.index], c=label_ful['tVal'], s=25, cmap='coolwarm')
plt.colorbar(sc, label='Trend t-value')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
plt.xticks(rotation=45)
plt.title('Trend Scanning Labels (Full Period)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.tight_layout()
plt.savefig('vis/Trend_Scanning_Labels.png')
plt.close()
print("✓ Trend scanning labels chart saved to vis/Trend_Scanning_Labels.png")






### Model Development
print("Starting model development...")


# Prepare data for models
X_all = prices[feature_cols].copy().dropna()
common_index = X_all.index.intersection(label_ful.index)
X_all = X_all.loc[common_index]
y_all = label_ful.loc[common_index, 'bin']

# Convert labels from {-1, 1} to {0, 1}
y_all = (y_all == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, shuffle=False)
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

### Model Selection (Random Forest)
print("Training Random Forest model...")

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

# Create time series cross-validation strategy
tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for speed

# Use RandomizedSearchCV for Random Forest
rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_SEED),
    param_distributions=rf_param_grid,
    n_iter=10,  # Reduced iterations for speed
    scoring='accuracy',
    cv=tscv,
    verbose=1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

# Fit the random search
rf_search.fit(X_train, y_train)

# Get the best parameters
rf_best_params = rf_search.best_params_
print("Best Random Forest parameters:")
for param, value in rf_best_params.items():
    print(f"  {param}: {value}")

# Train Random Forest with best parameters
rf_model = RandomForestClassifier(**rf_best_params, random_state=RANDOM_SEED)
rf_model.fit(X_train, y_train)
print("✓ Random Forest trained successfully")

### Model Selection (XGBoost)
print("Training XGBoost model...")

# Define parameter grid for XGBoost
xgb_param_dist = {
    'n_estimators': [80, 100],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.07],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.1],
    'reg_lambda': [1.0]
}

# xgb_base = xgb.XGBClassifier(...)
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=RANDOM_SEED,
    n_jobs=-1
)

# Time series cross-validation
xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_dist,
    n_iter=10,  # Reduced iterations for speed
    scoring='accuracy',
    cv=tscv,
    verbose=1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

# Fit the hyperparameter search on X_train and y_train
xgb_search.fit(X_train, y_train)

# Get best parameters
xgb_best_params = xgb_search.best_params_
print("Best XGBoost parameters:")
for param, value in xgb_best_params.items():
    print(f"  {param}: {value}")

# Train XGBoost model
xgb_model = xgb.XGBClassifier(**xgb_best_params, objective='binary:logistic', eval_metric='logloss',
                              random_state=RANDOM_SEED, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# Compute metrics (moved to evaluation section)
print("✓ XGBoost model trained successfully")

### Model Selection (LSTM replacing MLP)
print("Training LSTM model...")

# Normalize features for LSTM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM: (samples, timesteps, features)
sequence_length = 10  # Reduced sequence length for speed
X_train_lstm = np.array([X_train_scaled[i:i + sequence_length] for i in range(len(X_train_scaled) - sequence_length)])
y_train_lstm = y_train[sequence_length:]
X_test_lstm = np.array([X_test_scaled[i:i + sequence_length] for i in range(len(X_test_scaled) - sequence_length)])
y_test_lstm = y_test[sequence_length:]

# Define LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))  # Simplified architecture
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train LSTM model
lstm_model = create_lstm_model((sequence_length, X_train.shape[1]))
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Early stopping for speed
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping],
               verbose=1)  # Reduced epochs and increased batch size
print("✓ LSTM model trained successfully")

### Feature Importance for LSTM
print("Computing feature importance for LSTM model...")

# Modified function to handle 3D input for LSTM
def compute_lstm_feature_importance(model, X_test, y_test, sequence_length, n_repeats=10):
    """
    Compute permutation importance for an LSTM model with 3D input data.

    Parameters:
    - model: Trained LSTM model
    - X_test: Scaled test features (2D array)
    - y_test: Test labels
    - sequence_length: Number of timesteps in each sequence
    - n_repeats: Number of times to permute each feature

    Returns:
    - importances: Array of feature importance scores
    """
    # Prepare the test data as sequences
    X_test_seq = np.array([X_test[i:i + sequence_length] for i in range(len(X_test) - sequence_length)])
    y_test_seq = y_test[sequence_length:]

    # Get the number of features
    n_features = X_test_seq.shape[2]

    # Initialize an array to hold importance scores
    importances = np.zeros((n_features,))

    # Compute baseline accuracy
    baseline_pred = (model.predict(X_test_seq) > 0.5).astype(int).flatten()
    baseline_accuracy = accuracy_score(y_test_seq, baseline_pred)
    print(f"Baseline accuracy computed: {baseline_accuracy:.4f}")

    # Iterate over each feature
    for feature in range(n_features):
        scores = []
        print(f"Permuting feature {feature + 1}/{n_features}...")
        for _ in range(n_repeats):
            # Create a copy of X_test_seq
            X_permuted = X_test_seq.copy()
            # Permute the feature across all time steps
            for t in range(sequence_length):
                np.random.shuffle(X_permuted[:, t, feature])
            # Predict with permuted data
            permuted_pred = (model.predict(X_permuted) > 0.5).astype(int).flatten()
            # Compute accuracy
            permuted_accuracy = accuracy_score(y_test_seq, permuted_pred)
            # Compute importance as the drop in accuracy
            importance = baseline_accuracy - permuted_accuracy
            scores.append(importance)
        # Average the importance scores for this feature
        importances[feature] = np.mean(scores)
        print(f"Feature {feature + 1} importance: {importances[feature]:.4f}")

    print("Permutation importance calculation completed.")
    return importances

# Compute feature importance for LSTM
lstm_feature_importances = compute_lstm_feature_importance(lstm_model, X_test_scaled, y_test, sequence_length)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_cols, lstm_feature_importances)
plt.title('Feature Importance - LSTM (Permutation Importance)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('vis/Feature_Importance_LSTM.png')
plt.close()
print("✓ LSTM feature importance chart saved to vis/Feature_Importance_LSTM.png")

### Training Strategies
print("Generating signals with different training strategies...")

# One-time prediction
def generate_signals_one_time(model, X_test):
    print("Starting one-time prediction...")
    # Map the classification label from {0,1} to {−1,+1} (handled later in alignment)
    test_preds = model.predict(X_test)
    print("One-time prediction completed. Sample signals:")
    print(pd.Series(test_preds, index=X_test.index, name='signal_one_time').head())
    return pd.Series(test_preds, index=X_test.index, name='signal_one_time')

# Rolling Window
"""
Rolling-window on hourly data.
• Train on hourly slices of length train_window,
• Test on the subsequent test_window,
• Then roll forward by step_size and repeat.
Concatenate all test-slice predictions into one long signal series.

Returns a Series of signals in {−1, +1}, indexed at each test hour.
"""
def generate_signals_rolling(model_class, param_dict, X_all, y_all, train_window=400, test_window=100, step_size=50):
    idx_all = X_all.index.sort_values()
    N = len(idx_all)
    signals = []
    signal_times = []
    start = 0
    iteration = 0
    while True:
        train_start = start
        train_end = min(start + train_window, N)  # Avoid index errors
        test_end = min(train_end + test_window, N)
        if test_end > N or train_end >= N:
            break
        iteration += 1
        print(
            f"Rolling Window - Iteration {iteration}: Training window {train_start}-{train_end}, Testing window {train_end}-{test_end}")
        train_idx = idx_all[train_start:train_end]
        test_idx = idx_all[train_end:test_end]
        X_train_slice = X_all.loc[train_idx]
        y_train_slice = y_all.loc[train_idx]
        m = model_class(**param_dict)
        m.fit(X_train_slice, y_train_slice)
        X_test_slice = X_all.loc[test_idx]
        test_preds = m.predict(X_test_slice)
        signals.extend(test_preds.tolist())
        signal_times.extend(test_idx.tolist())
        start += step_size  # Increased step size for speed
    result = pd.Series(signals, index=pd.DatetimeIndex(signal_times), name='signal_rolling')
    # Handle duplicate timestamps by keeping the last prediction
    print(f"Before removing duplicates, sig_rolling shape: {result.shape}")
    result = result.groupby(result.index).last()
    print(f"After removing duplicates, sig_rolling shape: {result.shape}")
    print("Rolling window signals generated. Sample signals:")
    print(result.head())
    return result

# Expanding Window
"""
Generates expanding-window signals on hourly data, but only retrains every
`retrain_every` hours and (optionally) uses only the last `max_train_window`
hours of history for each retrain.
Returns:
  pd.Series of signals in {−1, +1}, indexed from `test_start` to `X_all.index[-1]`.
"""
def generate_signals_expanding(model_class, param_dict, X_all, y_all, initial_train_hours, test_start, retrain_every=4,
                               max_train_window=None):
    signals = []
    signal_times = []
    idx_all = X_all.index.sort_values()
    start_pos = idx_all.get_loc(test_start)
    last_model = None
    last_fit_pos = None
    iteration = 0
    for pos in range(start_pos, len(idx_all)):
        t = idx_all[pos]
        # Determine whether we need to retrain at this pos:
        # Never fitted yet → we must fit now
        # If we've moved forward by retrain_every hours since last fit, then fit
        if last_model is None or (pos - last_fit_pos if last_fit_pos is not None else 0) >= retrain_every:
            iteration += 1
            # Determine the slice of historical hours to train on
            # Only keep the last max_train_window hours before 't'
            start_train_idx = 0 if max_train_window is None else max(0, pos - max_train_window)
            train_idx = idx_all[start_train_idx:pos]
            print(
                f"Expanding Window - Iteration {iteration}: Training window {start_train_idx}-{pos}, Current time {t}")
            X_train_slice = X_all.loc[train_idx]
            y_train_slice = y_all.loc[train_idx]
            m = model_class(**param_dict)
            m.fit(X_train_slice, y_train_slice)
            last_model = m
            last_fit_pos = pos
        # Use last_model to predict the signal at hour t
        feat_t = X_all.loc[[t]]
        test_pred = last_model.predict(feat_t)[0]
        signals.append(test_pred)
        signal_times.append(t)
    # Not enough history to fit yet → skip (flat = 0) or continue until enough hours pass (handled by start_pos)
    result = pd.Series(signals, index=pd.DatetimeIndex(signal_times), name='signal_expanding')
    # Verify no duplicates (should be unique due to sequential prediction)
    assert not result.index.duplicated().any(), "sig_expanding has duplicate indices"
    print("Expanding window signals generated. Sample signals:")
    print(result.head())
    return result

### Signal Generation and Backtesting Setup
# Generate the signals
test_start = X_test.index[0]
sig_onetime = generate_signals_one_time(xgb_model, X_test)
print("✓ One-time prediction signals generated")
sig_rolling = generate_signals_rolling(xgb.XGBClassifier, xgb_model.get_params(), X_all, y_all, train_window=400,
                                       test_window=100, step_size=50)
print("✓ Rolling window signals generated")
sig_expanding = generate_signals_expanding(xgb.XGBClassifier, xgb_model.get_params(), X_all, y_all,
                                           initial_train_hours=len(X_train), test_start=test_start, retrain_every=4,
                                           max_train_window=None)
print("✓ Expanding window signals generated")
print("✓ Signal generation completed")

# align_signals_to_returns
"""
Given signals indexed by timestamps (hourly) and returns (hourly),
forward-fill any missing signal values (especially before test_start) with 0.
"""
def align_signals_to_returns(signals, returns):
    s_final = signals.reindex(returns.index).ffill().fillna(0).astype(int)
    # Map {0,1} → {−1,+1} if needed
    s_final = s_final.replace({0: -1, 1: 1})
    return s_final

# Compute strategy returns: s_{t-1} * r_t, then restrict to test_start onward
asset_returns = prices['close'].pct_change().fillna(0.0)
sig_one_final = align_signals_to_returns(sig_onetime, asset_returns)
sig_rolling_final = align_signals_to_returns(sig_rolling, asset_returns)
sig_expanding_final = align_signals_to_returns(sig_expanding, asset_returns)

strat_ret_one = (sig_one_final.shift(1) * asset_returns).loc[test_start:]
strat_ret_rolling = (sig_rolling_final.shift(1) * asset_returns).loc[test_start:]
strat_ret_expanding = (sig_expanding_final.shift(1) * asset_returns).loc[test_start:]

strategy_one = (1 + strat_ret_one).cumprod().rename('One-Time Prediction Strategy')
strategy_rolling = (1 + strat_ret_rolling).cumprod().rename('Rolling Window Strategy')
strategy_expanding = (1 + strat_ret_expanding).cumprod().rename('Expanding Window Strategy')

# Buy & Hold benchmark on hourly data:
bh_equity = (1 + asset_returns.loc[test_start:]).cumprod().rename('Buy and Hold')
bh_equity.iloc[0] = 1.0

# Plot the equity curves
plt.figure(figsize=(12, 6))
plt.plot(strategy_one.index, strategy_one.values, label='One-Time Prediction', linewidth=1.5)
plt.plot(strategy_rolling.index, strategy_rolling.values, label='Rolling Window', linewidth=1.5)
plt.plot(strategy_expanding.index, strategy_expanding.values, label='Expanding Window', linewidth=1.5)
plt.plot(bh_equity.index, bh_equity.values, label='Buy and Hold', linewidth=1.5, linestyle='--')
plt.legend(loc='best')
plt.title('Equity Curves of Strategies vs. Buy and Hold')
plt.xlabel('Timestamp (Hourly)')
plt.ylabel('Normalized Equity (Initial=1.0)')
plt.grid(True)
plt.tight_layout()
plt.savefig('vis/Equity_Curves.png')
plt.close()
print("✓ Equity curves chart saved to vis/Equity_Curves.png")

### Feature Importance and Model Evaluation
print("Starting model evaluation...")

# Evaluate your model's performance on the test period using classification metrics (accuracy, precision, recall, F1-score) and confusion matrix analysis.
### Model Evaluation Function
"""Comprehensive model evaluation"""
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, is_lstm=False):
    # Make predictions
    if is_lstm:
        X_train_seq = np.array([X_train[i:i + sequence_length] for i in range(len(X_train) - sequence_length)])
        X_test_seq = np.array([X_test[i:i + sequence_length] for i in range(len(X_test) - sequence_length)])
        train_pred = (model.predict(X_train_seq) > 0.5).astype(int).flatten()
        test_pred = (model.predict(X_test_seq) > 0.5).astype(int).flatten()
        train_proba = model.predict(X_train_seq).flatten()
        test_proba = model.predict(X_test_seq).flatten()
        y_train_adj = y_train[sequence_length:]
        y_test_adj = y_test[sequence_length:]
    else:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        y_train_adj = y_train
        y_test_adj = y_test

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train Accuracy': accuracy_score(y_train_adj, train_pred),
        'Test Accuracy': accuracy_score(y_test_adj, test_pred),
        'Train Precision': precision_score(y_train_adj, train_pred),
        'Test Precision': precision_score(y_test_adj, test_pred),
        'Train Recall': recall_score(y_train_adj, train_pred),
        'Test Recall': recall_score(y_test_adj, test_pred),
        'Train F1 Score': f1_score(y_train_adj, train_pred),
        'Test F1 Score': f1_score(y_test_adj, test_pred),
        'Train AUC': roc_auc_score(y_train_adj, train_proba),
        'Test AUC': roc_auc_score(y_test_adj, test_proba),
        'test_pred': test_pred,
        'test_proba': test_proba
    }
    cm = confusion_matrix(y_test_adj, test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1'])
    metrics['Confusion Matrix'] = cm_df
    cls_report = classification_report(y_test_adj, test_pred, labels=[0, 1], target_names=['Class 0', 'Class 1'])
    metrics['Classification Report'] = cls_report
    # Store predictions for ROC curve
    return metrics

# Evaluate models and collect results
rf_results = evaluate_model(rf_model, X_train, y_train, X_test, y_test, 'Random Forest')
xgb_results = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, 'XGBoost')
lstm_results = evaluate_model(lstm_model, X_train_scaled, y_train, X_test_scaled, y_test, 'LSTM', is_lstm=True)

# Print out key metrics:
print("\nRandom Forest Evaluation Results:")
print(f"Test Accuracy: {rf_results['Test Accuracy']:.4f}")
print(f"Test Precision: {rf_results['Test Precision']:.4f}")
print(f"Test Recall: {rf_results['Test Recall']:.4f}")
print(f"Test F1 Score: {rf_results['Test F1 Score']:.4f}")
print(f"Test AUC: {rf_results['Test AUC']:.4f}")
# Show confusion matrix:
print("Confusion Matrix:\n", rf_results['Confusion Matrix'])
# Show the full classification report:
print("Classification Report:\n", rf_results['Classification Report'])

print("\nXGBoost Evaluation Results:")
print(f"Test Accuracy: {xgb_results['Test Accuracy']:.4f}")
print(f"Test Precision: {xgb_results['Test Precision']:.4f}")
print(f"Test Recall: {xgb_results['Test Recall']:.4f}")
print(f"Test F1 Score: {xgb_results['Test F1 Score']:.4f}")
print(f"Test AUC: {xgb_results['Test AUC']:.4f}")
print("Confusion Matrix:\n", xgb_results['Confusion Matrix'])
print("Classification Report:\n", xgb_results['Classification Report'])

print("\nLSTM Evaluation Results:")
print(f"Test Accuracy: {lstm_results['Test Accuracy']:.4f}")
print(f"Test Precision: {lstm_results['Test Precision']:.4f}")
print(f"Test Recall: {lstm_results['Test Recall']:.4f}")
print(f"Test F1 Score: {lstm_results['Test F1 Score']:.4f}")
print(f"Test AUC: {lstm_results['Test AUC']:.4f}")
print("Confusion Matrix:\n", lstm_results['Confusion Matrix'])
print("Classification Report:\n", lstm_results['Classification Report'])

# Visualize confusion matrices
plt.figure(figsize=(15, 5))
for i, (results, name) in enumerate([(rf_results, 'Random Forest'), (xgb_results, 'XGBoost'), (lstm_results, 'LSTM')],
                                    1):
    plt.subplot(1, 3, i)
    sns.heatmap(results['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('vis/Confusion_Matrices.png')
plt.close()
print("✓ Confusion matrices chart saved to vis/Confusion_Matrices.png")

# Visualize ROC curves
plt.figure(figsize=(10, 6))
for results, name in [(rf_results, 'Random Forest'), (xgb_results, 'XGBoost'), (lstm_results, 'LSTM')]:
    fpr, tpr, _ = roc_curve(y_test[sequence_length:] if name == 'LSTM' else y_test, results['test_proba'])
    auc = roc_auc_score(y_test[sequence_length:] if name == 'LSTM' else y_test, results['test_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves of Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('vis/ROC_Curves.png')
plt.close()
print("✓ ROC curves chart saved to vis/ROC_Curves.png")

# Visualize feature importance (RF and XGBoost only)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
rf_importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.bar(rf_importances.index, rf_importances.values)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.bar(xgb_importances.index, xgb_importances.values)
plt.title('Feature Importance - XGBoost')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('vis/Feature_Importance.png')
plt.close()
print("✓ Feature importance chart saved to vis/Feature_Importance.png")

print("✓ Model evaluation completed")

### Backtesting with Backtrader
print("Starting backtesting performance analysis...")

# Create a Backtrader datafeed.
prices = prices.sort_index()
prices.index = pd.to_datetime(prices.index)
prices = prices[['open', 'high', 'low', 'close', 'volume']]
# timeframe=bt.TimeFrame.Minutes, compression=60  # 1 hour bars
data = bt.feeds.PandasData(dataname=prices, timeframe=bt.TimeFrame.Minutes, compression=60)

# Define a generic strategy class that works with all models
class ModelStrategy(bt.Strategy):
    params = dict(
        model=None,  # The trained model (RF, XGBoost, or LSTM)
        features=None,  # X_all DataFrame (hourly features, indexed by timestamp)
        stake=1,  # Number of units to trade
        conf_threshold=0.65,  # Confidence threshold lowered to 0.65
        trade_every=12,  # Trade every 12 hours
        stop_loss=0.05  # 5% stop loss
    )

    def __init__(self):
        self.idx_all = self.p.features.index.sort_values()
        self.order = None
        self.dataclose = self.datas[0].close
        self.entry_price = None
        self.position_type = None  # 'long' or 'short'

    def notify_order(self, order):
        # This is called whenever an order changes state
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.datetime(0)
            if order.isbuy():
                print(
                    f"[{dt.isoformat()}] Buy executed, Price: {order.executed.price:.4f}, Cost: {order.executed.value:.4f}")
                self.entry_price = order.executed.price
                self.position_type = 'long'
            elif order.issell():
                print(
                    f"[{dt.isoformat()}] Sell executed, Price: {order.executed.price:.4f}, Cost: {order.executed.value:.4f}")
                self.entry_price = order.executed.price
                self.position_type = 'short'
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"[{self.datas[0].datetime.datetime(0).isoformat()}] Order canceled/margin insufficient/rejected")
            self.order = None

    def notify_trade(self, trade):
        # This is called whenever a trade is closed (or updated)
        if trade.isclosed:
            dt = self.datas[0].datetime.datetime(0)
            print(f"[{dt.isoformat()}] Trade closed - Gross P&L: {trade.pnl:.4f}, Net P&L: {trade.pnlcomm:.4f}")

    def next(self):
        # Current bar's datetime
        curr_dt = self.datas[0].datetime.datetime(0)
        # If this timestamp is not in our features index, do nothing
        if curr_dt not in self.idx_all:
            return
        # Find the integer position of this datetime
        pos = self.idx_all.get_loc(curr_dt)
        # Only trade every `trade_every` hours (skip intermediate bars)
        if pos % self.p.trade_every != 0:
            return

        # Stop-loss check
        if self.position.size != 0 and self.entry_price is not None:
            current_price = self.dataclose[0]
            if self.position_type == 'long' and (
                    self.entry_price - current_price) / self.entry_price > self.p.stop_loss:
                print(f"[{curr_dt.isoformat()}] Stop-loss triggered - Closing long @ {current_price:.4f}")
                self.close()
                self.entry_price = None
                self.position_type = None
            elif self.position_type == 'short' and (
                    current_price - self.entry_price) / self.entry_price > self.p.stop_loss:
                print(f"[{curr_dt.isoformat()}] Stop-loss triggered - Closing short @ {current_price:.4f}")
                self.close()
                self.entry_price = None
                self.position_type = None

        # Get features for current timestamp
        feat_curr = self.p.features.loc[[curr_dt]]

        # Predict using the model
        if isinstance(self.p.model, Sequential):  # LSTM model
            if pos >= sequence_length:
                seq_start = pos - sequence_length + 1
                seq_end = pos + 1
                X_seq = self.p.features.iloc[seq_start:seq_end].values
                X_seq_scaled = scaler.transform(X_seq)
                X_seq_reshaped = X_seq_scaled.reshape(1, sequence_length, -1)
                proba = self.p.model.predict(X_seq_reshaped, verbose=0)[0][0]
                signal = 1 if proba >= self.p.conf_threshold else -1 if proba <= (1 - self.p.conf_threshold) else 0
            else:
                signal = 0  # Not enough history for sequence
        else:  # Tree-based models (RF, XGBoost)
            proba = self.p.model.predict_proba(feat_curr)[0][1]
            signal = 1 if proba >= self.p.conf_threshold else -1 if proba <= (1 - self.p.conf_threshold) else 0

        # Trading logic
        pos_size = self.getposition(self.datas[0]).size
        if signal == 1 and pos_size <= 0:
            if self.order:
                self.cancel(self.order)
            if pos_size < 0:
                self.order = self.close()
            else:
                entry_price = self.dataclose[0]
                self.order = self.buy(size=self.p.stake)
                print(f"[{curr_dt.isoformat()}] Submitting buy order @ {entry_price:.4f}")
        elif signal == -1 and pos_size >= 0:
            if self.order:
                self.cancel(self.order)
            if pos_size > 0:
                self.order = self.close()
            else:
                entry_price = self.dataclose[0]
                self.order = self.sell(size=self.p.stake)
                print(f"[{curr_dt.isoformat()}] Submitting sell order @ {entry_price:.4f}")

# Function to run backtest for a given model
def run_backtest(model, model_name):
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(ModelStrategy, model=model, features=X_all, stake=1, conf_threshold=0.65, trade_every=12,
                        stop_loss=0.05)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Minutes, compression=60)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn', timeframe=bt.TimeFrame.Minutes, compression=60)

    print(f"Running backtest for {model_name}...")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    sharpe_info = strat.analyzers.sharpe.get_analysis()
    drawdown_info = strat.analyzers.drawdown.get_analysis()
    trade_info = strat.analyzers.trades.get_analysis()
    timereturn = strat.analyzers.timereturn.get_analysis()

    # Debugging: Print predicted probabilities for XGBoost
    if model_name == "XGBoost":
        probas = [model.predict_proba(X_all.loc[[dt]])[0][1] for dt in X_all.index]
        print(f"XGBoost Predicted Probabilities - Min: {min(probas):.4f}, Max: {max(probas):.4f}")

    # Calculate backtest metrics
    days = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = (final_value / 10000.0) ** (1 / days) - 1 if days > 0 else 0
    volatility = np.std(list(timereturn.values())) * np.sqrt(252) if timereturn else 0
    sharpe_ratio = sharpe_info.get('sharperatio', None)  # Default to None if not available
    downside_returns = [r for r in timereturn.values() if r < 0]
    sortino_ratio = sharpe_ratio if not downside_returns else (
            sharpe_ratio * np.std(list(timereturn.values())) / np.std(downside_returns)) if sharpe_ratio is not None else 0
    max_drawdown = drawdown_info.get('max', {}).get('drawdown', 0)
    avg_holding_period = trade_info.get('len', {}).get('average', 0) if trade_info else 0

    print(f"\n{model_name} Backtest Results:")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"CAGR: {cagr:.4f}")
    print(f"Volatility: {volatility:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio if sharpe_ratio is not None else 'N/A'}")
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Average Holding Period: {avg_holding_period:.2f} hours")
    print(f"✓ Backtesting for {model_name} completed")

# Run backtests for all models
run_backtest(rf_model, "Random Forest")
run_backtest(xgb_model, "XGBoost")
run_backtest(lstm_model, "LSTM")

# Close log file
sys.stdout = original_stdout
log_file.close()