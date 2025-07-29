import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

def get_annual_returns_old():
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

def etf_index():
    etf_classes = {
        "MSCI World": "URTH",
        "S&P500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Stoxx 600": "^STOXX",
        "Emerging Markets": "EEM",
        "Or": "GLD",
        "Obligations": "AGG",
        "Private Equity": "BX",
        "Bitcoin": "BTC-EUR",
        "Ethereum": "ETH-EUR",
        "Altcoins": "SOL-EUR"
    }
    return etf_classes
@st.cache_data
def get_annual_returns():
    etf_classes =etf_index()
    # {
    #     "MSCI World": "URTH",
    #     "S&P500": "^GSPC",
    #     "Nasdaq": "^IXIC",
    #     "Stoxx 600": "^STOXX",
    #     "Emerging Markets": "EEM",
    #     "Or": "GLD",
    #     "Obligations": "AGG",
    #     "Private Equity": "BX",
    #     "Bitcoin": "BTC-EUR",
    #     "Ethereum": "ETH-EUR",
    #     "Altcoins": "SOL-EUR"
    # }

    annual_returns = {}
    default_const_return=4

    for asset_class, ticker in etf_classes.items():
        try:
            data = yf.download(ticker, start="1930-01-01", progress=False, auto_adjust=True)
            if data.empty:
                annual_returns[asset_class] = default_const_return
                continue

            # Résample annuel sur les prix de clôture ajustés
            annual_prices = data['Close'].resample('Y').last()
            yearly_returns = annual_prices.pct_change().dropna()

            if yearly_returns.empty:
                annual_returns[asset_class] = default_const_return
                continue

            # Quantile 50% (médiane) en pourcentage
            median_return = np.percentile(yearly_returns.values, 50) * 100
            annual_returns[asset_class] = round(median_return, 2)

        except Exception as e:
            annual_returns[asset_class] = default_const_return

    return annual_returns

#
# if __name__ == '__main__':
#     returns = get_annual_returns()
#     for k, v in returns.items():
#         print(f"{k}: {v}%")
    #print(returns["S&P500"].iloc[0])