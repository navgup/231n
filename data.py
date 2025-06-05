import yfinance as yf
import mplfinance as mpf
import pandas as pd
import os
from polygon import RESTClient, exceptions
from datetime import datetime, timedelta, timezone
import time


client = RESTClient()   # picks up POLYGON_API_KEY env var

def _one_day(ticker: str, day: datetime) -> pd.DataFrame:
    """
    Fetch a single US‑trading day of 5‑minute bars from Polygon.
    `day` should be midnight‑UTC of desired date.
    """
    date_str = day.strftime("%Y-%m-%d")
    bars = client.get_aggs(
        ticker=ticker,
        multiplier=5,
        timespan="minute",
        from_=date_str,
        to=date_str,
        limit=50000,      # plenty for one day
        sort="asc",
        adjusted=True,
    )
    if not bars:
        return pd.DataFrame()
    records = [{
        "timestamp": b.timestamp / 1000,  # ms → s
        "open": b.open,
        "high": b.high,
        "low":  b.low,
        "close": b.close,
        "volume": b.volume,
    } for b in bars]
    df = pd.DataFrame.from_records(records).set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="s", utc=True
                              ).tz_convert("America/New_York")
    df.rename(columns=str.title, inplace=True)  # Open High Low Close Volume
    return df


def fetch_5m_polygon_full(ticker: str,
                          start_date: str,
                          end_date: str,
                          slice_days: int = 60,
                          sleep_sec: float = 0.25) -> pd.DataFrame:
    """
    Pull continuous 5‑minute bars from `start_date` (inclusive) up to
    `end_date` (exclusive) in `slice_days`‑day chunks.
    Dates must be YYYY‑MM‑DD strings (Eastern or UTC is fine).
    """
    # Convert to datetime (midnight UTC for consistency)
    end   = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    frames = []
    cursor_end = end                            # moving window upper bound

    while cursor_end > start:
        cursor_start = max(start, cursor_end - timedelta(days=slice_days))

        # Polygon expects strings
        from_str = cursor_start.strftime("%Y-%m-%d")
        to_str   = cursor_end.strftime("%Y-%m-%d")

        print(f"Fetching {from_str} → {to_str} …")
        try:
            bars = client.get_aggs(ticker, 5, "minute", from_=from_str, to=to_str,
                                   limit=50000, sort="asc", adjusted=True)
        except exceptions.BadResponse as e:
            # If plan doesn’t include this timeframe, log & break loop
            if "NOT_AUTHORIZED" in str(e):
                # skip this slice and move cursor further back
                cursor_end = cursor_start
                continue
            else:
                raise  # unknown error
        except Exception as e:
            cursor_end = cursor_start
            continue

        if bars:  # could be empty if slice is entirely weekend/holiday range
            records = [{
                "timestamp": b.timestamp / 1000,
                "open":  b.open,
                "high":  b.high,
                "low":   b.low,
                "close": b.close,
                "volume": b.volume,
            } for b in bars]

            df = (
                pd.DataFrame.from_records(records)
                .set_index("timestamp")
                .pipe(lambda d: d if d.empty else
                      d.assign(
                          datetime=pd.to_datetime(d.index, unit="s", utc=True)
                                  .tz_convert("America/New_York"))
                      .set_index("datetime")
                      .rename(columns=str.title))
            )
            frames.append(df)



        # move the cursor leftward
        cursor_end = cursor_start

    # concat / sort / de‑dup
    if not frames:
        return pd.DataFrame()
    all_bars = pd.concat(frames).sort_index().drop_duplicates()
    # trim to regular RTH once more
    all_bars = all_bars.between_time("09:30", "16:00")
    return all_bars



def save_candlestick_chart(data_window, save_path, size=(224, 224)):
    # Clean up to remove axes, grid, etc.
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=False)

    mpf.plot(data_window,
             type='candle',
             style=s,
             axisoff=True,
             figsize=(size[0]/100, size[1]/100),  # Convert px to inch for dpi=100
             tight_layout=True,
             savefig=dict(fname=save_path, dpi=100, pad_inches=0.0))

def generate_dataset(data, window_size=30, stride=15, output_dir='candles', ticker='SPY'):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(data) - window_size, stride):
        window = data.iloc[i:i+window_size]
        timestamp = window.index[-1].strftime('%Y%m%d_%H%M')
        filename = f'{ticker}_{timestamp}.png'
        save_path = os.path.join(output_dir, filename)
        save_candlestick_chart(window, save_path)

THRESH_PCT = 0.005

from uuid import uuid4


def generate_classification_dataset(df: pd.DataFrame, output_dir="clf_ds", *,
                                    window=30, horizon=5, stride=15,
                                    thresh: float = THRESH_PCT, ticker="SPY") -> str:
    os.makedirs(output_dir, exist_ok=True)
    #records: List[dict] = []
    records = []
    for i in range(0, len(df)-window-horizon, stride):
        win = df.iloc[i:i+window]
        # skip if gaps
        if len(win) != window:
            continue
        # Skip if window spans more than one calendar day (prevents gaps)
        if win.index[0].date() != win.index[-1].date():
            continue
        current_close = win['Close'].iloc[-1]
        future_close  = df['Close'].iloc[i+window+horizon-1]
        ret = (future_close - current_close) / current_close
        label = 2 if ret >  thresh else 0 if ret < -thresh else 1
        fname = f"{ticker}_{uuid4().hex}.png"; path=os.path.join(output_dir,fname)
        save_candlestick_chart(win, path)
        records.append({"path":path,"label":label})
    csv_path = os.path.join(output_dir,"labels.csv")
    pd.DataFrame(records).to_csv(csv_path,index=False)
    return csv_path

    
def main():
    # data = fetch_ohlcv_data('SPY', interval='5m', period='6mo')    
    tickers = [
    #Broad‑market ETFs
    # "IWM", "DIA", "VTI", "TQQQ"
    # "TQQQ"
    
    # # Sector ETFs (State Street “XL” family)
    # "XLK",  # Technology
    # "XLF",  # Financials
    # "XLV",  # Health Care
    # "XLE",  # Energy
    # "XLY",  # Consumer Discretionary
    # "XLP",  # Consumer Staples
    # "XLI",  # Industrials
    # "XLB",  # Materials
    # "XLRE", # Real Estate
    # "XLC",  # Communication Services
    
    # High‑volume single stocks
    "AAPL", "MSFT", "AMZN", "GOOGL",
    "TSLA", "META", "JPM", "NFLX", 
    
    # # Rate / volatility proxies
    # "TLT",  # 20‑Year Treasury ETF
    # "HYG",  # High‑Yield Corporate Bond ETF
    # "VXX",  # VIX Short‑Term Futures ETN
     
    # Leveraged ETFs (optional spice)
    # "TQQQ", "SQQQ", "UPRO", "SPXU"
]
    
    # data = fetch_5m_polygon_full_multi(tickers, end_date='2025-06-02', start_date='2025-04-14', slice_days=60, sleep_sec=15)
    # generate_classification_dataset_multi(data, output_dir='multi_classification')
    
    for ticker in tickers:
        print(f"Fetching {ticker}")
        output_dir = f"{ticker}_classification"
        data = fetch_5m_polygon_full(ticker, end_date='2025-06-02', start_date='2020-06-07', slice_days=30, sleep_sec=0) 
        generate_classification_dataset(data, output_dir=output_dir, ticker=ticker)
    # data = fetch_5m_polygon_full('SPY', end_date='2025-06-04', start_date='2020-06-07', slice_days=30, sleep_sec=0)
    # generate_classification_dataset(data, output_dir='SPY_classification_2', ticker='SPY')


if __name__ == '__main__':
    main()


# def fetch_5m_polygon_full_multi(tickers,
#                           start_date: str,
#                           end_date: str,
#                           slice_days: int = 60,
#                           sleep_sec: float = 0.3) -> pd.DataFrame:
#     """Return a **single dataframe** concatenating 5‑min bars for every
#     ticker requested (adds a 'Ticker' column).  Skips slices that trigger
#     NOT_AUTHORIZED.  Continues with other tickers.
#     """
#     if isinstance(tickers, str):
#         tickers = [tickers]

#     frames = []
#     for symbol in tickers:
#         end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
#         begin = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
#         cursor_end = end
#         while cursor_end > begin:
#             cursor_start = max(begin, cursor_end - timedelta(days=slice_days))
#             f_str, t_str = cursor_start.strftime("%Y-%m-%d"), cursor_end.strftime("%Y-%m-%d")
#             print(f"Fetching {symbol} {f_str} → {t_str} …")
#             try:
#                 bars = client.get_aggs(symbol, 5, "minute", from_=f_str, to=t_str,
#                                        limit=50000, sort="asc", adjusted=True)
#             except exceptions.BadResponse as e:
#                 if "NOT_AUTHORIZED" in str(e):
#                     print("failed")
#                     cursor_end = cursor_start; continue
#                 else:
#                     raise
#             except Exception as e:
#                 print("failed")
#                 cursor_end = cursor_start; continue

#             if bars:
#                 records = [{"timestamp": b.timestamp/1000, "open": b.open, "high": b.high,
#                             "low": b.low, "close": b.close, "volume": b.volume} for b in bars]
#                 df = (pd.DataFrame.from_records(records)
#                         .set_index("timestamp")
#                         .assign(Ticker=symbol)
#                         .pipe(lambda d: d if d.empty else d.assign(
#                               datetime=pd.to_datetime(d.index, unit="s", utc=True)
#                                        .tz_convert("America/New_York")))
#                         .set_index("datetime"))
#                 frames.append(df)
#             time.sleep(sleep_sec)
#             cursor_end = cursor_start
#     if not frames:
#         raise RuntimeError("No data fetched – check API key/limits")
#     full = pd.concat(frames).sort_index().drop_duplicates()
#     return full.between_time("09:30", "16:00")

# def generate_classification_dataset_multi(df: pd.DataFrame, output_dir="clf_ds", *,
#                                           window=60, horizon=5, stride=10,
#                                           thresh: float = THRESH_PCT) -> str:
#     os.makedirs(output_dir, exist_ok=True)
#     rows = []
#     # group by ticker so we respect continuity within each symbol
#     for ticker, sub in df.groupby("Ticker"):
#         print(df.columns)
#         sub = sub.sort_index()
#         closes = sub['Close']
#         for i in range(0, len(sub) - window - horizon, stride):
#             win = sub.iloc[i:i+window]
#             if win.index[0].date() != win.index[-1].date():
#                 continue
#             future_close = closes.iloc[i+window+horizon-1]
#             ret = (future_close - win['Close'].iloc[-1]) / win['Close'].iloc[-1]
#             label = 2 if ret > thresh else 0 if ret < -thresh else 1
#             fname = f"{ticker}_{uuid4().hex}.png"; path=os.path.join(output_dir,fname)
#             save_candlestick_chart(win, path)
#             rows.append({"path":path, "label":label})
#     csv_path = os.path.join(output_dir, "labels.csv")
#     pd.DataFrame(rows).to_csv(csv_path, index=False)
#     return csv_path