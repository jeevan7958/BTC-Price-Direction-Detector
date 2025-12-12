import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Crypto Forecaster", layout="wide")

st.title("₿ Bitcoin Price Direction Predictor")
st.markdown("Powered by **Transformer Architecture** & **Stationary Returns Analysis**")

# --- 1. LOAD RESOURCES (Cached for Speed) ---
@st.cache_resource
def load_resources():
    # Define paths relative to app.py
    # Assume app.py is run from the project root (python -m streamlit run app/app.py)
    # OR from inside the app folder. Check both.
    
    scaler_path = 'data/processed/scaler.pkl'
    model_path = 'models/transformer_model.keras'
    
    # Check if files exist, if not, try adjusting path (for deployment compatibility)
    if not os.path.exists(scaler_path):
        # Maybe we are inside the 'app' folder?
        scaler_path = '../data/processed/scaler.pkl'
        model_path = '../models/transformer_model.keras'

    # Load Scaler
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        scaler = None
    
    # Load Model
    try:
        model = tf.keras.models.load_model(model_path)
    except (FileNotFoundError, OSError):
        model = None
        
    return scaler, model

scaler, model = load_resources()

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "BTC-USD")

# SLIDER UPDATE: Range from 30 days up to 5000 days (approx 13 years)
days_back = st.sidebar.slider("History to Show (Days)", 30, 5000, 365)

# --- 3. LIVE DATA FETCH ---
st.subheader(f"Analyzing {ticker}...")

def get_data(ticker):
    # DOWNLOAD UPDATE: Fetch 'max' history instead of just '3mo'
    df = yf.download(ticker, period="max", interval="1d")
    
    # Fix for MultiIndex columns in new yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

try:
    with st.spinner("Fetching live market data..."):
        df = get_data(ticker)
    
    # DYNAMIC CHART: Filter data based on the slider
    display_df = df.tail(days_back)
    st.line_chart(display_df['Close'])
    
    # --- 4. PREPARE DATA FOR AI ---
    # Calculate Stationary Returns (Same math as Week 2)
    df['Returns'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Handle infinite values just in case
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Get the last 14 days of data for the prediction window
    last_14_days = df.tail(14).copy()
    
    if len(last_14_days) < 14:
        st.error("Not enough recent data to make a prediction.")
    else:
        # Prepare Input Features
        features = last_14_days[['Returns', 'Volume_Change']].values
        
        # Scale (using the loaded scaler)
        if scaler:
            scaled_features = scaler.transform(features)
        else:
            # Fallback if scaler missing: simplistic normalization
            scaled_features = features 
            st.warning("Scaler file not found. Using raw data (Accuracy reduced).")

        # Reshape for Transformer: (1 sample, 14 time steps, 2 features)
        model_input = scaled_features.reshape(1, 14, 2)
        
        # --- 5. PREDICT ---
        col1, col2, col3 = st.columns(3)
        current_price = df['Close'].iloc[-1]
        
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")

        # LOGIC: Check if we have a real model or need to use Demo Mode
        if model:
            # 1. Get the Raw Prediction (e.g., 0.372)
            raw_prediction = model.predict(model_input)
            raw_value = raw_prediction[0][0]
            
            # 2. INVERSE TRANSFORM (The Translation Step)
            # The scaler expects 2 features [Return, Volume], so we create a dummy array
            dummy_array = np.zeros((1, 2))
            dummy_array[0, 0] = raw_value  # Put prediction in the 'Return' slot
            
            # Translate back to real world percentage
            real_prediction = scaler.inverse_transform(dummy_array)[0, 0]
            
            # 3. Use the REAL prediction for display
            next_price = current_price * (1 + real_prediction)
            
            # ... inside the 'if model:' block, after calculating real_prediction ...
            
            with col2:
                if real_prediction > 0:
                    st.metric("AI Prediction", "BULLISH 🚀", delta=f"{real_prediction*100:.2f}%")
                else:
                    st.metric("AI Prediction", "BEARISH 📉", delta=f"{real_prediction*100:.2f}%", delta_color="inverse")
                
                # NEW: Add Context
                momentum_14d = df['Close'].pct_change(14).iloc[-1]
                st.caption(f"14-Day Momentum: {momentum_14d*100:.2f}%")
                if momentum_14d > 0 and real_prediction > 0:
                     st.caption("👉 The model is following the strong 2-week uptrend.")
                elif momentum_14d > 0 and real_prediction < 0:
                     st.caption("👉 The model predicts a reversal despite the uptrend.")
            
            with col3:
                st.metric("Predicted Next Close", f"${next_price:,.2f}")
                
        else:
            # DEMO MODE (Fallback)
            st.info("⚠️ Model file not found. Running in Demo Mode (Momentum Strategy).")
            
            # Simple momentum logic for demo
            avg_return = np.mean(features[:, 0])
            next_price = current_price * (1 + avg_return)
            
            with col2:
                if avg_return > 0:
                    st.metric("Trend Prediction", "UPWARD ↗️", delta=f"{avg_return*100:.2f}%")
                else:
                    st.metric("Trend Prediction", "DOWNWARD ↘️", delta=f"{avg_return*100:.2f}%", delta_color="inverse")
            
            with col3:
                st.metric("Est. Next Close", f"${next_price:,.2f}")

except Exception as e:
    st.error(f"Error processing data: {e}")
    st.write("Debug info:", e)

st.markdown("---")
st.caption("Disclaimer: This is a research project utilizing Transformer Neural Networks. Past performance is not indicative of future results.")