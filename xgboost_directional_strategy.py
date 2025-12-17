# xgboost_directional_strategy.py

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb

# ---------------------------
# 1. Data Extraction (replace with SQL extraction in prod)
# ---------------------------
# Example: Daily US Equity data (AAPL)
symbol = "AAPL"
start_date = "2019-01-01"
end_date = "2021-12-31"
data = yf.download(symbol, start=start_date, end=end_date)
data['Return'] = data['Adj Close'].pct_change()

# ---------------------------
# 2. Feature Engineering
# ---------------------------
# 20-day EMA
data['EMA20'] = data['Adj Close'].ewm(span=20, adjust=False).mean()

# RSI calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data['Adj Close'])

# Lagged returns
data['Return_1'] = data['Return'].shift(1)
data['Return_2'] = data['Return'].shift(2)
data['Return_3'] = data['Return'].shift(3)

# Drop NaNs
data.dropna(inplace=True)

# ---------------------------
# 3. Target Construction (Binary)
# ---------------------------
data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)

# ---------------------------
# 4. Train/Test Split
# ---------------------------
features = ['EMA20', 'RSI', 'Return_1', 'Return_2', 'Return_3']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------------------------
# 5. XGBoost Model
# ---------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ---------------------------
# 6. Predictions & Evaluation
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Directional Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# ---------------------------
# 7. Optional: Backtesting Framework
# ---------------------------
data['Signal'] = model.predict(data[features])
data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
data['Cumulative_Market'] = (1 + data['Return']).cumprod()

# Plot cumulative returns
import matplotlib.pyplot as plt
plt.plot(data['Cumulative_Strategy'], label='Strategy')
plt.plot(data['Cumulative_Market'], label='Market')
plt.legend()
plt.show()
