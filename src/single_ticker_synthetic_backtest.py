#=======================
# synthetic_backtest.py
#=======================
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from spread_constructor import (
    construct_best_spread_for_chain,
    BullCallSpread,
)

risk_free_rate = 0.03
target_DTE = 60
realized_vol_window = 20
ticker = "AAPL"
start_date = "2021-01-01"
end_date = "2025-12-31"

#==== Black-Scholes Call Pricer ======#
from math import log, sqrt, exp
from math import erf

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price for a European call option
    S = spot price
    K = strike
    T = time to expiry in years
    r = risk-free rate (annualized)
    sigma = volatility (annualized)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    call_price = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    return call_price

#==== Realized Volatility (IV Proxy) ======#
def get_history_with_realized_IV(
    ticker: str,
    start: str,
    end: str,
    window: int = realized_vol_window
) -> pd.DataFrame:
    """
    Downloads price history and computes realized-volatility-based IV proxy (rv_iv).
    Robust to yfinance returning either normal columns or MultiIndex columns.
    Guarantees returned df has columns: Close, rv_iv (plus intermediates).
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No price history for {ticker} in {start}...{end}")

    df = df.copy()

    # ---- Handle possible MultiIndex columns (e.g. ('Close','INTC')) ----
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            close = df[("Adj Close", ticker)].rename("Close")
        elif ("Close", ticker) in df.columns:
            close = df[("Close", ticker)].rename("Close")
        else:
            # Fallback: try any column where level-0 is 'Adj Close' or 'Close'
            lvl0 = df.columns.get_level_values(0)
            if "Adj Close" in lvl0:
                close = df.loc[:, lvl0 == "Adj Close"].iloc[:, 0].rename("Close")
            elif "Close" in lvl0:
                close = df.loc[:, lvl0 == "Close"].iloc[:, 0].rename("Close")
            else:
                raise KeyError(f"Could not find Close/Adj Close in yfinance columns: {df.columns}")

        df = pd.DataFrame({"Close": close})

    else:
        # Normal single-level columns
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        elif "Close" not in df.columns:
            raise KeyError(f"yfinance output missing Close and Adj Close. Columns: {df.columns}")

        df = df[["Close"]].copy()

    # Ensure datetime index and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # log returns
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    # rolling std of daily returns
    df["sigma_daily"] = df["log_ret"].rolling(window=window).std()
    # annualize -> IV proxy
    df["rv_iv"] = df["sigma_daily"] * np.sqrt(252)
    # clip to reasonable range [15%, 150%]
    df["rv_iv"] = df["rv_iv"].clip(lower=0.15, upper=1.50)

    return df

#==== Synthetic Option Chain Generator ======#
def generate_synthetic_chain_for_date(
    S: float,
    iv: float,
    T_days: int,
    r: float = risk_free_rate,
) -> pd.DataFrame:

    if T_days <= 0:
        raise ValueError("T days must be positive")

    T_years = T_days / 365.0

    if S < 50:
        step = 1.0
    elif S < 200:
        step = 2.0
    else:
        step = 5.0

    # Align strikes to strike grid so strikes look realistic (no weird float strikes)
    k_min = max(1.0, np.floor((0.5 * S) / step) * step)
    k_max = np.ceil((1.5 * S) / step) * step
    strikes = np.arange(k_min, k_max + step, step)

    rows = []
    for K in strikes:
        mid = bs_call_price(S, K, T_years, r, iv)

        # skip super tiny calls
        if mid < 0.05:
            continue

        spread_factor = 0.02  # 2% bid-ask half spread
        bid = mid * (1 - spread_factor)
        ask = mid * (1 + spread_factor)

        # enforce some minimum spread
        if ask - bid < 0.05:
            mid_center = (bid + ask) / 2
            bid = mid_center - 0.025
            ask = mid_center + 0.025

        rows.append({
            "strike": float(K),
            "bid": float(bid),
            "ask": float(ask),
            "impliedVolatility": float(iv),
        })

    return pd.DataFrame(rows)

# ====== All Spread Trades ===========

def run_synthetic_backtest_roll(
    ticker: str,
    start: str = start_date,
    end: str = end_date,
    target_dte: int = target_DTE,
    step_after_expiry_days: int = 1,
) -> pd.DataFrame:
    """
    Runs a non-overlapping synthetic bull call spread backtest:
    enter -> hold to expiry (~target_dte) -> re-enter after expiry -> repeat.

    Always returns a DataFrame (may be empty). Never returns None.
    """
    df = get_history_with_realized_IV(ticker, start, end)

    # need IV available, so drop rows before realized vol window fills
    df = df.dropna(subset=["rv_iv", "Close"]).copy()
    if df.empty:
        raise ValueError("No usable rows after RV/IV calculation (maybe window too large or data missing).")

    # helper to snap a datetime to the last available trading day <= dt
    def snap_to_trading_day(dt: datetime) -> datetime | None:
        ts = pd.Timestamp(dt)
        prev = df.index[df.index <= ts]
        if len(prev) == 0:
            return None
        return prev[-1].to_pydatetime()

    # helper to snap to the first trading day >= dt (for re-entry after expiry)
    def snap_forward_to_trading_day(dt: datetime) -> datetime | None:
        ts = pd.Timestamp(dt)
        nxt = df.index[df.index >= ts]
        if len(nxt) == 0:
            return None
        return nxt[0].to_pydatetime()

    entry_dt = snap_forward_to_trading_day(datetime.fromisoformat(start))
    end_dt = datetime.fromisoformat(end)

    trades: list[dict] = []

    # IMPORTANT: guard against None
    while entry_dt is not None and entry_dt <= end_dt:
        entry_ts = pd.Timestamp(entry_dt)
        if entry_ts not in df.index:
            # Snap forward if somehow not in index (should be rare)
            entry_dt = snap_forward_to_trading_day(entry_dt + timedelta(days=1))
            continue

        S = float(df.loc[entry_ts, "Close"])
        iv = float(df.loc[entry_ts, "rv_iv"])

        # target expiry + snap to last trading day <= target
        raw_expiry = entry_dt + timedelta(days=target_dte)
        expiry_dt = snap_to_trading_day(raw_expiry)
        if expiry_dt is None:
            break  # no expiry available in data

        dte = (expiry_dt - entry_dt).days
        if dte <= 0:
            entry_dt = snap_forward_to_trading_day(entry_dt + timedelta(days=1))
            continue

        # generate synthetic chain at entry
        chain = generate_synthetic_chain_for_date(S, iv, dte, r=risk_free_rate)
        if chain.empty:
            entry_dt = snap_forward_to_trading_day(entry_dt + timedelta(days=1))
            continue

        expiry_str = expiry_dt.date().isoformat()

        spread = construct_best_spread_for_chain(
            ticker=ticker,
            expiry=expiry_str,
            chain=chain,
            underlying_price=S,
            dte=dte,
        )

        if spread is None:
            # no spread fits constraints; skip forward a day
            entry_dt = snap_forward_to_trading_day(entry_dt + timedelta(days=1))
            continue

        # realized underlying at expiry
        expiry_ts = pd.Timestamp(expiry_dt)
        if expiry_ts not in df.index:
            # should not happen due to snap_to_trading_day, but guard anyway
            expiry_dt = snap_to_trading_day(expiry_dt)
            if expiry_dt is None:
                break
            expiry_ts = pd.Timestamp(expiry_dt)

        S_T = float(df.loc[expiry_ts, "Close"])

        long_intr = max(0.0, S_T - spread.long_strike)
        short_intr = max(0.0, S_T - spread.short_strike)
        spread_intr = long_intr - short_intr
        spread_intr = max(0.0, min(spread_intr, spread.width))

        pnl = spread_intr - spread.debit
        ret_on_debit = pnl / spread.debit if spread.debit > 0 else np.nan
        ret_on_width = pnl / spread.width if spread.width > 0 else np.nan
        
        trades.append({
            "ticker": ticker,
            "entry_date": entry_dt.date().isoformat(),
            "expiry_date": expiry_dt.date().isoformat(),
            "dte": dte,
            "S_entry": S,
            "iv_entry": iv,
            "K1_long": spread.long_strike,
            "K2_short": spread.short_strike,
            "width": spread.width,
            "debit": spread.debit,
            "S_expiry": S_T,
            "intrinsic_expiry": spread_intr,
            "pnl": pnl,
            "ret_on_debit": ret_on_debit,
            "ret_on_width": ret_on_width,
            "rrr": getattr(spread, "rrr", np.nan),
            "max_profit": getattr(spread, "max_profit", np.nan),
        })

        # roll forward: next entry after expiry
        next_entry_raw = expiry_dt + timedelta(days=step_after_expiry_days)
        if next_entry_raw > end_dt:
            break

        next_entry_dt = snap_forward_to_trading_day(next_entry_raw)
        if next_entry_dt is None:
            break

        entry_dt = next_entry_dt

    # ALWAYS return a DataFrame (never None)
    return pd.DataFrame(trades)


# ====== Main: run INTC rolling trades and output ====== #
if __name__ == "__main__":
    results = run_synthetic_backtest_roll(
        ticker=ticker,
        start=start_date,
        end=end_date,
        target_dte=target_DTE,
        step_after_expiry_days=1,
    )

    print("\n=== Synthetic Rolling Backtest: INTC (~60 DTE Bull Call Spreads) ===")
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Target DTE: {target_DTE}")
    print(f"Trades found: {len(results)}\n")

    if results.empty:
        print("No trades constructed under current constraints.")
    else:
        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 160)
        print(results.head(15).to_string(index=False))

        out_path = f"synthetic_trades_{ticker}_{start_date}_to_{end_date}_DTE{target_DTE}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nSaved trades to: {out_path}")

