import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical data from Yahoo Finance and saves it to CSV.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # CHECK: Handle MultiIndex columns (a common issue with recent yfinance versions)
    # If columns look like ('Close', 'BTC-USD'), flatten them to just 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Check if data is empty
    if data.empty:
        print(f"Error: No data found for {ticker}. Check your ticker symbol or dates.")
        return None

    # Reset index to make 'Date' a proper column (easier for CSVs)
    data.reset_index(inplace=True)
    
    # Display info
    print(f"Success! Retrieved {len(data)} rows.")
    print(data.head())
    
    return data

def save_data(data, ticker):
    """
    Saves the dataframe to the data/raw directory.
    """
    # Create filename
    filename = f"data/raw/{ticker}_historical.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    TICKER = "BTC-USD"  # Change to 'AAPL' or 'GOOG' as needed
    START = "2020-01-01"
    END = "2025-01-01"
    
    # --- EXECUTION ---
    df = fetch_data(TICKER, START, END)
    if df is not None:
        save_data(df, TICKER)