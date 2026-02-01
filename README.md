Features to download from yfinance:
1. Core - 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
2. Price to use - Adj Close (accounts for splits/dividends)
3. Target - next day return (preferably log return)
4. Exogenous features: Time series outside the stock that might help predict it. 
    Core features from Ticker:
    1. Return & Momentum (The most important):
        - logret_1d = log(AdjClose_t / AdjClose_{t-1})
        - ret_1d = AdjClose_t / AdjClose_{t-1} - 1
        Lags (this is your core "time-series" input):
        - logret_1d_lag_1 .... logret_1d_lag_60 (paper uses 60-step lookback)
        Rolling momentum (means of returns):
        - ret_mean_5, ret_mean_10, ret_mean_20, ret_mean_60 (Why use these? Check it out)
    2. Volatility (Critical for stock data):
        Rolling volatility (std dev of returns):
        - ret_vol_5, ret_vol_10, ret_vol_20, ret_vol_60
        Range-based volatility proxies:
        - hl_range = (High - Low) / Close
        - oc_change = (Close - Open) / Open
    3. Trend Indicators: 
        Compute on Adj Close
        Moving Averages:
        - sma_5, sma_10, sma_20, sma_60
        Spreads (trend strength):
        - sma_spread_5 = AdjClose - sma_5 - 1
        - sma_spread_20 = AdjClose - sma_20 - 1
        - sma_spread_60 = AdjClose - sma_60 - 1
        Exponential Moving Averages:
        - ema_12, ema_26
        MACD family:
        - macd = ema_12 - ema_26
        - macd_signal = ema(macd, 9)
        - macd_hist = macd - macd_signal
    4. Mean reversion / "overbought-oversold":
        - rsi_14 (Relative Strength Index)
        Bollinger Bands (20, 2sigma):
        - bb_mid_20, bb_upper_20, bb_lower_20, bb_width_20
    5. Volume (captures participation)
        - vol_change = Volume_t / Volume_{t-1} - 1
        - vol_z_20 = (Volume_t - mean(Volume_{t-1 to t-20})) / std(Volume_{t-1 to t-20})
        - obv (On-Balance Volume)
5. Cross-stock "peer" exogenous features: For each ticker T, include the other three tickers' returns as exogenous inputs
    - if T = AAPL, peers = {MSFT, GOOG, AMZN}, then peer_logret_lag_1 ... peer_logret_lag_20 for each peer
6. Market/sector/volatility exogenous features:
    1. Market & tech-sector exposure:
        - SPY (S&P 500 ETF) - spy_logret_lag_1 ... spy_logret_lag_20
        - XLK (Technology Select Sector SPDR Fund) - xlk_logret_lag_1 ... qqq_logret_lag_20
        - QQQ (tracks Nasdaq-100) - qqq_logret_lag_1 ... qqq_logret_lag_20
    2. Volatility / risk sentiment:
        ^VIX (VIX Index) - use daily % change and logs  -> Leading measure of expected near-term volatility derived from S&P 500 options
7. Calendar Features
    - dow (day of week 0-4)
    - month (1-12)
    - is_month_end (0/1) (check what is what correctly)
8. Lags & Windows:
    - Endogenous return lags: 1 ... 60 
    - Exogenous lags : 1 ... 20 
    - Rolling windows: 5, 10, 20, 60