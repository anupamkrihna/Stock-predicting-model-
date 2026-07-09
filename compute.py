import os
import json
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- CONFIGURATION ---
STOCK_TICKER = "ITC.NS"        # Change this to any Yahoo Finance ticker
MACRO_TICKER = "^NSEI"        # Nifty 50 index as our macro landscape landscape
LOOKBACK_DAYS = 365 * 3       # 3 years of historical data

def run_prediction_pipeline():
    print(f"🚀 Starting Layered Quant Pipeline for {STOCK_TICKER}...")
    
    # 1. FETCH DATA
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=LOOKBACK_DAYS)
    
    stock_df = yf.download(STOCK_TICKER, start=start_date, end=end_date)
    macro_df = yf.download(MACRO_TICKER, start=start_date, end=end_date)
    
    # Clean multi-index columns if present in newer yfinance versions
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    if isinstance(macro_df.columns, pd.MultiIndex):
        macro_df.columns = macro_df.columns.get_level_values(0)

    # Sync dataframes
    df = pd.DataFrame(index=stock_df.index)
    df['Stock_Close'] = stock_df['Close']
    df['Stock_Volume'] = stock_df['Volume']
    df['Macro_Close'] = macro_df['Close']
    df = df.dropna()
    
    df['Stock_Return'] = df['Stock_Close'].pct_change()
    df['Macro_Return'] = df['Macro_Close'].pct_change()
    df = df.dropna()

    # --- LAYER 1: GARCH & MONTE CARLO ---
    print("📈 Layer 1: Running GARCH and Monte Carlo...")
    # Fit GARCH(1,1) to model dynamic risk
    garch = arch_model(df['Stock_Return'] * 100, vol='Garch', p=1, q=1, dist='Normal')
    garch_res = garch.fit(disp='off')
    
    # Predict tomorrow's volatility (scaled back down)
    next_day_vol = np.sqrt(garch_res.forecast(horizon=1).variance.iloc[-1, 0]) / 100
    current_price = float(df['Stock_Close'].iloc[-1])
    
    # Monte Carlo Simulations (1,000 paths, 5 days out)
    sim_paths = 1000
    sim_days = 5
    drift = float(df['Stock_Return'].mean())
    
    mc_results = np.zeros((sim_days, sim_paths))
    for i in range(sim_paths):
        prices = [current_price]
        for d in range(sim_days):
            # Price path driven by drift + dynamic GARCH volatility shock
            shock = np.random.normal(drift, next_day_vol)
            prices.append(prices[-1] * (1 + shock))
        mc_results[:, i] = prices[1:]
        
    # Extract probabilistic features for the next trading day
    day_1_sims = mc_results[0, :]
    mc_upper_band = float(np.percentile(day_1_sims, 95))
    mc_lower_band = float(np.percentile(day_1_sims, 5)) # Value at Risk proxy

    # Add dynamic historical volatility to dataframe for Layer 2
    df['GARCH_Vol'] = garch_res.conditional_volatility / 100

    # --- LAYER 2: MULTIPLE REGRESSION (Macro Landscape) ---
    print("🌐 Layer 2: Computing Macroeconomic Regression Baseline...")
    X_reg = df[['Macro_Return', 'GARCH_Vol', 'Stock_Volume']].iloc[:-1]
    y_reg = df['Stock_Close'].shift(-1).dropna()
    
    # Drop any remaining alignment NaNs
    valid_idx = X_reg.dropna().index.intersection(y_reg.index)
    X_reg = X_reg.loc[valid_idx]
    y_reg = y_reg.loc[valid_idx]

    reg_model = LinearRegression()
    reg_model.fit(X_reg, y_reg)
    
    # Compute the baseline macro-anchored price history
    df['Macro_Baseline'] = reg_model.predict(df[['Macro_Return', 'GARCH_Vol', 'Stock_Volume']])

    # --- LAYER 3: LSTM DEEP LEARNING (Pattern Sequencer) ---
    print("🤖 Layer 3: Training Sequential LSTM Engine...")
    # The LSTM learns from the variation between actual prices and our macro baseline
    df['Residual'] = df['Stock_Close'] - df['Macro_Baseline']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_residuals = scaler.fit_transform(df[['Residual']])
    
    # Build lookback sequences (past 20 days to predict next step)
    lookback = 20
    X_lstm, y_lstm = [], []
    for i in range(lookback, len(scaled_residuals)):
        X_lstm.append(scaled_residuals[i-lookback:i, 0])
        y_lstm.append(scaled_residuals[i, 0])
        
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    
    # Quick, lightweight LSTM network
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(lookback, 1)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_lstm, y_lstm, epochs=8, batch_size=16, verbose=0)
    
    # Predict tomorrow's residual variation
    last_sequence = scaled_residuals[-lookback:]
    last_sequence = np.reshape(last_sequence, (1, lookback, 1))
    pred_scaled_residual = model.predict(last_sequence, verbose=0)
    pred_residual = float(scaler.inverse_transform(pred_scaled_residual)[0][0])
    
    # Final Layered Prediction = Tomorrow's Macro Baseline + LSTM Adjusted Pattern
    tomorrow_macro_input = np.array([[df['Macro_Return'].iloc[-1], df['GARCH_Vol'].iloc[-1], df['Stock_Volume'].iloc[-1]]])
    tomorrow_baseline = reg_model.predict(tomorrow_macro_input)[0]
    final_lstm_prediction = float(tomorrow_baseline + pred_residual)

    # --- SAVE OUTPUTS FOR DASHBOARD ---
    recent_history = df.tail(30)
    output_data = {
        "ticker": STOCK_TICKER,
        "update_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "current_price": round(current_price, 2),
        "garch_volatility": round(float(next_day_vol) * 100, 2),
        "mc_upper": round(mc_upper_band, 2),
        "mc_lower": round(mc_lower_band, 2),
        "lstm_prediction": round(final_lstm_prediction, 2),
        "historical_dates": recent_history.index.strftime("%Y-%m-%d").tolist(),
        "historical_prices": recent_history['Stock_Close'].round(2).tolist(),
        "macro_baseline": recent_history['Macro_Baseline'].round(2).tolist()
    }
    
    with open("predictions.json", "w") as f:
        json.dump(output_data, f, indent=4)
        
    print("✅ Success! 'predictions.json' generated perfectly.")

if __name__ == "__main__":
    run_prediction_pipeline()