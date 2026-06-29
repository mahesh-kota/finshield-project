import yfinance as yf

with open("data/stock_universe.txt") as f:
    tickers = [line.strip() for line in f if line.strip()]

print(f"Testing {len(tickers)} tickers...\n")

failed = []
for t in tickers[:10]:  # test first 10 only for speed
    data = yf.download(t, period="5d", progress=False)
    if data.empty:
        failed.append(t)
        print(f"FAILED: {t}")
    else:
        print(f"OK: {t} - {len(data)} rows")

print(f"\n{len(failed)} failed out of 10 tested")
if failed:
    print("Failed tickers:", failed)
