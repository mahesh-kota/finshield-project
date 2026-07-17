import yfinance as yf

# NSE stocks use the .NS suffix on Yahoo Finance
ticker = "RELIANCE.NS"

data = yf.download(ticker, start="2024-01-01", end="2024-06-01")
print(data.head())
print("\nTotal rows fetched:", len(data))
print("\nColumns:", list(data.columns))
