import yfinance as yf
import pandas as pd
from datetime import datetime

def get_annual_returns():
    etf_classes = {
        "MSCI World": "URTH",               # iShares MSCI World ETF
        "S&P500": "^GSPC",                  # S&P 500 Index
        "Nasdaq": "^IXIC",                  # NASDAQ Composite
        "Stoxx 600": "^STOXX",              # STOXX Europe 600 Index
        "Emerging Markets": "EEM",          # iShares MSCI Emerging Markets ETF
        "Or": "GLD",                        # SPDR Gold Shares
        "Obligations": "AGG",              # iShares Core US Aggregate Bond ETF
        "Private Equity": "BX",             # Blackstone as proxy for PE
        "Bitcoin":"BTC-EUR",               #btc eur
        "Ethereum":"ETH-EUR",
        "Altcoins":"SOL-EUR"             #solana/doge as benchmark?
    }

    annual_returns = {}

    for asset_class, ticker in etf_classes.items():
        try:
            data = yf.download(ticker, start="1985-01-01", progress=False)
            if data.empty:
                annual_returns[asset_class] = None
                continue

            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]

            start_date = data.index[0]
            end_date = data.index[-1]
            years = (end_date - start_date).days / 365.25

            cagr = (end_price / start_price) ** (1 / years) - 1
            annual_returns[asset_class] = round(cagr * 100, 2)
        except Exception as e:
            annual_returns[asset_class] = 5

    return annual_returns

#
# if __name__ == '__main__':
#     returns = get_annual_returns()
#     for k, v in returns.items():
#         print(f"{k}: {v}%")
    #print(returns["S&P500"].iloc[0])