import numpy as np
import pandas as pd
import yfinance as yf


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def download_yahoo(symbols: str | list[str], start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:

    tickers = [symbols] if isinstance(symbols, str) else list(symbols)

    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbols}.")

  
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(map(str, df.columns.get_level_values(0)))
        lvl1 = set(map(str, df.columns.get_level_values(1)))
        field_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "open", "high", "low", "close", "adj close", "volume"}

        if len(lvl0 & field_names) >= 3:
            
            df = df.swaplevel(0, 1, axis=1)
        

        
        new_cols = []
        for t, f in df.columns:
            f2 = str(f).strip().lower()
            if f2 == "adj close":
                f2 = "adj_close"
            new_cols.append((str(t), f2))
        df.columns = pd.MultiIndex.from_tuples(new_cols, names=["ticker", "field"])
    else:
        
        sym = tickers[0]
        cols = [str(c).strip().lower() for c in df.columns]
        cols = ["adj_close" if c == "adj close" else c for c in cols]
        df.columns = pd.MultiIndex.from_product([[sym], cols], names=["ticker", "field"])

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def make_rolling_features(df: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.names != ["ticker", "field"]:
        raise ValueError("df must have MultiIndex columns (ticker, field) from download_yahoo().")

    tickers = df.columns.get_level_values("ticker").unique().tolist()

  
    mret = None
    if market is not None and not market.empty:
        if isinstance(market.columns, pd.MultiIndex):
           
            m_tickers = market.columns.get_level_values("ticker").unique().tolist()
            m0 = m_tickers[0]
            if (m0, "adj_close") in market.columns:
                m_adj = market[(m0, "adj_close")]
            else:
                m_adj = market[(m0, "close")]
        else:
            
            m_adj = market["adj_close"] if "adj_close" in market.columns else market["close"]
        mret = m_adj.pct_change()

    feats_by_ticker: dict[str, pd.DataFrame] = {}

    for t in tickers:
        tdf = df[t]

        o = tdf["open"]
        h = tdf["high"]
        l = tdf["low"]
        c = tdf["close"]
        v = tdf["volume"].astype(float)
        adj = tdf["adj_close"] if "adj_close" in tdf.columns else c

        out = pd.DataFrame(index=df.index)

        
        out["ret_1"] = adj.pct_change()
        out["logret_1"] = np.log(adj).diff()
        out["hl_range"] = (h - l) / (c.replace(0, np.nan))
        out["oc_gap"] = (o - c.shift(1)) / (c.shift(1).replace(0, np.nan))
        out["co_move"] = (c - o) / (o.replace(0, np.nan))
        out["vol_chg"] = v.pct_change()
        out["dollar_vol"] = c * v

        
        win_returns = [2, 3, 5, 10, 14, 20, 30, 60]
        win_vol = [5, 10, 20, 30, 60]
        win_z = [10, 20, 60]
        win_volu = [5, 10, 20, 60]

        
        for w in win_returns:
            out[f"ret_{w}"] = adj.pct_change(w)
            out[f"logret_{w}"] = np.log(adj).diff(w)
            out[f"mom_{w}"] = adj / adj.shift(w) - 1

        
        for w in win_vol:
            out[f"rv_logret_{w}"] = out["logret_1"].rolling(w).std() * np.sqrt(252)
            out[f"vol_ret_{w}"] = out["ret_1"].rolling(w).std()
            out[f"mean_ret_{w}"] = out["ret_1"].rolling(w).mean()

        
        prev_close = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
        out["true_range"] = tr
        for w in [5, 10, 14, 20, 60]:
            out[f"atr_{w}"] = tr.rolling(w).mean()
            out[f"range_mean_{w}"] = out["hl_range"].rolling(w).mean()
            out[f"range_std_{w}"] = out["hl_range"].rolling(w).std()

        
        for w in win_volu:
            out[f"vol_sma_{w}"] = v.rolling(w).mean()
            out[f"vol_ema_{w}"] = v.ewm(span=w, adjust=False).mean()
            out[f"vol_z_{w}"] = (v - v.rolling(w).mean()) / (v.rolling(w).std().replace(0, np.nan))
            out[f"dvol_z_{w}"] = (out["dollar_vol"] - out["dollar_vol"].rolling(w).mean()) / (
                out["dollar_vol"].rolling(w).std().replace(0, np.nan)
            )

        
        for w in win_z:
            out[f"price_z_{w}"] = (adj - adj.rolling(w).mean()) / (adj.rolling(w).std().replace(0, np.nan))
            out[f"ret_z_{w}"] = (out["ret_1"] - out["ret_1"].rolling(w).mean()) / (
                out["ret_1"].rolling(w).std().replace(0, np.nan)
            )

        
        for p in [7, 14, 21]:
            out[f"rsi_{p}"] = _rsi(adj, p)

        
        ema12 = _ema(adj, 12)
        ema26 = _ema(adj, 26)
        macd = ema12 - ema26
        signal = _ema(macd, 9)
        out["macd"] = macd
        out["macd_signal"] = signal
        out["macd_hist"] = macd - signal

        
        mid = adj.rolling(20).mean()
        sd = adj.rolling(20).std()
        bb_up = mid + 2 * sd
        bb_dn = mid - 2 * sd
        out["bb_mid_20"] = mid
        out["bb_up_20"] = bb_up
        out["bb_dn_20"] = bb_dn
        out["bb_width_20"] = (bb_up - bb_dn) / (mid.replace(0, np.nan))
        out["bb_pos_20"] = (adj - bb_dn) / ((bb_up - bb_dn).replace(0, np.nan))

       
        if mret is not None:
            out["mkt_ret_1"] = mret
            out["beta_60"] = out["ret_1"].rolling(60).cov(mret) / (mret.rolling(60).var().replace(0, np.nan))
            out["corr_60"] = out["ret_1"].rolling(60).corr(mret)
            out["alpha_60"] = out["ret_1"] - out["beta_60"] * mret

        out = out.replace([np.inf, -np.inf], np.nan)
        feats_by_ticker[t] = out

    
    feats = pd.concat(feats_by_ticker, axis=1)
    feats.columns.names = ["ticker", "feature"]
    return feats


def make_windows(
    features: pd.DataFrame,
    adj_close: pd.DataFrame,
    window_len: int = 32,
    horizon: int = 1,
    target: str = "next_return",
    dropna: bool = True,
):
    if not isinstance(features.columns, pd.MultiIndex):
        raise ValueError("features must have MultiIndex columns (ticker, feature).")

    tickers = features.columns.get_level_values(0).unique().tolist()

    
    if isinstance(adj_close.columns, pd.MultiIndex):
        
        ac = {}
        for t in tickers:
            if (t, "adj_close") in adj_close.columns:
                ac[t] = adj_close[(t, "adj_close")]
            else:
                ac[t] = adj_close[(t, "close")]
        adj = pd.DataFrame(ac, index=adj_close.index)
    else:
        adj = adj_close.copy()

   
    feature_names = features.columns.get_level_values(1).unique().tolist()

    X_list, y_list, meta_rows = [], [], []

    for t in tickers:
        ft = features[t].copy()  
        y_series = adj[t].pct_change(horizon).shift(-horizon)
        if target == "next_updown":
            y_series = (y_series > 0).astype(float)
        else:
            y_series = y_series.astype(float)

        data = ft.copy()
        data["__y__"] = y_series

        if dropna:
            data = data.dropna()

        feat_cols = [c for c in data.columns if c != "__y__"]
        arrX = data[feat_cols].to_numpy(dtype=np.float32)
        arry = data["__y__"].to_numpy(dtype=np.float32)
        idx = data.index

        for i in range(window_len - 1, len(data)):
            X_list.append(arrX[i - window_len + 1 : i + 1])
            y_list.append(arry[i])
            meta_rows.append({"date": idx[i], "ticker": t})

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, window_len, len(feature_names)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    return X, y, feature_names, meta


if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "NVDA"]
    market_symbol = "^GSPC"

    df = download_yahoo(symbols, start="2016-01-01")
    mkt = download_yahoo(market_symbol, start="2016-01-01")

    feats = make_rolling_features(df, market=mkt)

  
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    for t in feats.columns.get_level_values("ticker").unique():
        
        df_t = feats[t].copy()

        
        df_t = df_t.reset_index().rename(columns={"index": "date"})

        out_path = os.path.join(out_dir, f"{t}_rolling.csv")
        df_t.to_csv(out_path, index=False)

        print(f"Saved {t} rolling features to {out_path}")

    X, y, feature_names, meta = make_windows(
        features=feats,
        adj_close=df,  
        window_len=64,
        horizon=1,
        target="next_return",
    )

    print("Tickers:", symbols)
    print("Features per day:", len(feature_names))
    print("X shape:", X.shape, "y shape:", y.shape)
    if len(meta):
        print("First sample:", meta.iloc[0].to_dict())
        print("Last sample:", meta.iloc[-1].to_dict())