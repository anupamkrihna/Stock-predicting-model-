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
# Add any Yahoo Finance tickers you want to track here!
STOCK_TICKERS = ["ITC.NS", "IOC.NS", "RELIANCE.NS", "TCS.NS", "SBIN.NS", "INFY.NS"]
MACRO_TICKER = "^NSEI"        # Nifty 50 Index as the macroeconomic landscape
LOOKBACK_DAYS = 365 * 3       # 3 years of historical data

def run_prediction_pipeline():
    print(f"🚀 Starting Multi-Stock Layered Quant Pipeline...")
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=LOOKBACK_DAYS)
    
    # Pre-fetch macro data to save time
    print(f"🌐 Fetching Macro Index: {MACRO_TICKER}")
    macro_raw = yf.download(MACRO_TICKER, start=start_date, end=end_date)
    if isinstance(macro_raw.columns, pd.MultiIndex):
        macro_raw.columns = macro_raw.columns.get_level_values(0)
        
    master_output = {
        "update_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stocks": {}
    }

    for ticker in STOCK_TICKERS:
        try:
            print(f"\nProcessing: {ticker} --------------------")
            stock_raw = yf.download(ticker, start=start_date, end=end_date)
            
            if stock_raw.empty or len(stock_raw) < 100:
                print(f"⚠️ Insufficient data for {ticker}, skipping.")
                continue
                
            if isinstance(stock_raw.columns, pd.MultiIndex):
                stock_raw.columns = stock_raw.columns.get_level_values(0)

            # Synchronize dataframes
            df = pd.DataFrame(index=stock_raw.index)
            df['Stock_Close'] = stock_raw['Close']
            df['Stock_Volume'] = stock_raw['Volume']
            df['Macro_Close'] = macro_raw['Close']
            df = df.dropna()
            
            df['Stock_Return'] = df['Stock_Close'].pct_change()
            df['Macro_Return'] = df['Macro_Close'].pct_change()
            df = df.dropna()

            # --- LAYER 1: GARCH & MONTE CARLO ---
            garch = arch_model(df['Stock_Return'] * 100, vol='Garch', p=1, q=1, dist='Normal')
            garch_res = garch.fit(disp='off')
            
            next_day_vol = np.sqrt(garch_res.forecast(horizon=1).variance.iloc[-1, 0]) / 100
            current_price = float(df['Stock_Close'].iloc[-1])
            
            sim_paths, sim_days = 1000, 5
            drift = float(df['Stock_Return'].mean())
            mc_results = np.zeros((sim_days, sim_paths))
            
            for i in range(sim_paths):
                prices = [current_price]
                for d in range(sim_days):
                    shock = np.random.normal(drift, next_day_vol)
                    prices.append(prices[-1] * (1 + shock))
                mc_results[:, i] = prices[1:]
                
            day_1_sims = mc_results[0, :]
            mc_upper_band = float(np.percentile(day_1_sims, 95))
            mc_lower_band = float(np.percentile(day_1_sims, 5))

            df['GARCH_Vol'] = garch_res.conditional_volatility / 100

            # --- LAYER 2: MULTIPLE REGRESSION ---
            X_reg = df[['Macro_Return', 'GARCH_Vol', 'Stock_Volume']].iloc[:-1]
            y_reg = df['Stock_Close'].shift(-1).dropna()
            valid_idx = X_reg.dropna().index.intersection(y_reg.index)
            
            reg_model = LinearRegression()
            reg_model.fit(X_reg.loc[valid_idx], y_reg.loc[valid_idx])
            df['Macro_Baseline'] = reg_model.predict(df[['Macro_Return', 'GARCH_Vol', 'Stock_Volume']])

            # --- LAYER 3: LSTM DEEP LEARNING ---
            df['Residual'] = df['Stock_Close'] - df['Macro_Baseline']
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_residuals = scaler.fit_transform(df[['Residual']])
            
            lookback = 20
            X_lstm, y_lstm = [], []
            for i in range(lookback, len(scaled_residuals)):
                X_lstm.append(scaled_residuals[i-lookback:i, 0])
                y_lstm.append(scaled_residuals[i, 0])
                
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
            
            model = Sequential([
                LSTM(32, return_sequences=False, input_shape=(lookback, 1)),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_lstm, y_lstm, epochs=6, batch_size=32, verbose=0)
            
            last_sequence = np.reshape(scaled_residuals[-lookback:], (1, lookback, 1))
            pred_residual = float(scaler.inverse_transform(model.predict(last_sequence, verbose=0))[0][0])
            
            tomorrow_macro = np.array([[df['Macro_Return'].iloc[-1], df['GARCH_Vol'].iloc[-1], df['Stock_Volume'].iloc[-1]]])
            tomorrow_baseline = reg_model.predict(tomorrow_macro)[0]
            final_lstm_prediction = float(tomorrow_baseline + pred_residual)

            # Calculate Expected Return Direction Edge
            edge_pct = ((final_lstm_prediction - current_price) / current_price) * 100
            signal = "BULLISH EDGE" if edge_pct > 0.5 else "BEARISH EDGE" if edge_pct < -0.5 else "NEUTRAL"

            # Package individual stock statistics
            recent_history = df.tail(30)
            master_output["stocks"][ticker] = {
                "current_price": round(current_price, 2),
                "garch_volatility": round(float(next_day_vol) * 100, 2),
                "mc_upper": round(mc_upper_band, 2),
                "mc_lower": round(mc_lower_band, 2),
                "lstm_prediction": round(final_lstm_prediction, 2),
                "edge_percent": round(edge_pct, 2),
                "signal": signal,
                "historical_dates": recent_history.index.strftime("%Y-%m-%d").tolist(),
                "historical_prices": recent_history['Stock_Close'].round(2).tolist(),
                "macro_baseline": recent_history['Macro_Baseline'].round(2).tolist()
            }
            print(f"✅ Successfully compiled analytical layers for {ticker}")

        except Exception as e:
            print(f"❌ Failed to process structure for {ticker}. Error: {e}")
            continue
            
    with open("predictions.json", "w") as f:
        json.dump(master_output, f, indent=4)
    print("\n🎉 Master 'predictions.json' dataset saved perfectly.")

if __name__ == "__main__":
    run_prediction_pipeline()
