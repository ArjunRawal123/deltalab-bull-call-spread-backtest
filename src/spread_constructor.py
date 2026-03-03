#========================
# spread_constructor.py
#========================
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# ----------------------------
# Config parameters
# ----------------------------
width_fraction = 0.08
width_tol_low = 0.50
width_tol_high = 1.50

MIN_ABS_WIDTH = 3.0

@dataclass
class BullCallSpread:
    ticker: str
    expiry: str
    underlying_price: float

    long_strike: float
    long_bid: float
    long_ask: float

    short_strike: float
    short_bid: float
    short_ask: float

    width: float
    debit: float
    max_profit: float  # MP = W - D
    rrr: float         # reward-to-risk ratio = MP / D


def get_underlying_price(ticker: str) -> float:
    tk = yf.Ticker(ticker)
    info = tk.history(period="1d")
    if info.empty:
        raise ValueError(f"No price data for {ticker}")
    return float(info["Close"].iloc[-1])


def get_calls_for_expiry(ticker: str, expiry: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry).calls
    if chain.empty:
        raise ValueError(f"No call options for {ticker} at expiry {expiry}")
    return chain.copy()


def choose_long_leg(chain: pd.DataFrame, underlying_price: float) -> pd.Series:
    """Slightly ITM long call; fallback to closest strike if no ITM."""
    itm = chain[chain["strike"] <= underlying_price].copy()
    if itm.empty:
        tmp = chain.copy()
        tmp["moneyness_diff"] = (tmp["strike"] - underlying_price).abs()
        return tmp.sort_values("moneyness_diff").iloc[0]

    itm["moneyness_diff"] = (underlying_price - itm["strike"]).abs()
    long_leg = itm.sort_values("moneyness_diff").iloc[0]
    return long_leg


def generate_short_leg_candidates_normalized(
    chain: pd.DataFrame,
    long_strike: float,
    underlying_price: float,
    width_fraction: float = width_fraction,
    tol_low: float = width_tol_low,
    tol_high: float = width_tol_high,
    min_abs_width: float = MIN_ABS_WIDTH,
) -> pd.DataFrame:
    """
    Short legs such that width ≈ width_fraction * S, within [tol_low, tol_high].
    """
    otm = chain[chain["strike"] > long_strike].copy()
    if otm.empty:
        return pd.DataFrame()

    w_target = max(width_fraction * underlying_price, min_abs_width)
    w_min = tol_low * w_target
    w_max = tol_high * w_target

    otm["width"] = otm["strike"] - long_strike
    candidates = otm[(otm["width"] >= w_min) & (otm["width"] <= w_max)].copy()

    if not candidates.empty:
        candidates["width_diff"] = (candidates["width"] - w_target).abs()
        candidates = candidates.sort_values("width_diff")

    return candidates


def allowed_debit_fraction(dte: int) -> float:
    """DTE-dependent max debit as fraction of width."""
    if dte >= 45:
        return 0.50
    elif dte >= 30:
        return 0.42
    else:
        return 0.36


def construct_best_spread_for_chain(
    ticker: str,
    expiry: str,
    chain: pd.DataFrame,
    underlying_price: float,
    dte: int,
) -> Optional[BullCallSpread]:

    long_leg = choose_long_leg(chain, underlying_price)

    long_strike = float(long_leg["strike"])
    long_bid = float(long_leg.get("bid", np.nan))
    long_ask = float(long_leg.get("ask", np.nan))

    shorts = generate_short_leg_candidates_normalized(chain, long_strike, underlying_price)
    if shorts.empty:
        return None

    max_debit_frac = allowed_debit_fraction(dte)

    best_spread: Optional[BullCallSpread] = None
    best_rrr = -np.inf

    for _, short in shorts.iterrows():
        short_strike = float(short["strike"])
        short_bid = float(short.get("bid", np.nan))
        short_ask = float(short.get("ask", np.nan))

        width = short_strike - long_strike

        # Basic sanity
        if width <= 0 or width < MIN_ABS_WIDTH:
            continue
        if np.isnan(long_ask) or np.isnan(short_bid):
            continue

        # Pay ask, receive bid
        debit = long_ask - short_bid
        if debit <= 0:
            continue

        # ---------- PROBABILITY-QUALITY FILTERS ----------
        breakeven_price = long_strike + debit
        breakeven_dist_pct = (breakeven_price - underlying_price) / underlying_price
        if breakeven_dist_pct > 0.08:
            continue

        short_dist_pct = (short_strike - underlying_price) / underlying_price
        if short_dist_pct > 0.10:
            continue

        debit_frac = debit / width
        if debit_frac < 0.20 or debit_frac > 0.55:
            continue

        # Enforce DTE debit cap
        if debit > max_debit_frac * width:
            continue
        # ---------- END FILTERS ----------

        max_profit = width - debit
        rrr = max_profit / debit

        if rrr > best_rrr:
            best_rrr = rrr
            best_spread = BullCallSpread(
                ticker=ticker,
                expiry=expiry,
                underlying_price=underlying_price,
                long_strike=long_strike,
                long_bid=float(long_bid) if not np.isnan(long_bid) else 0.0,
                long_ask=float(long_ask),
                short_strike=short_strike,
                short_bid=float(short_bid),
                short_ask=float(short_ask) if not np.isnan(short_ask) else 0.0,
                width=width,
                debit=debit,
                max_profit=max_profit,
                rrr=rrr,
            )

    return best_spread


def build_spread_for_ticker_and_expiry(
    ticker: str,
    expiry: str,
    dte: Optional[int] = None,
) -> Optional[BullCallSpread]:
    """
    Build a spread for a given ticker+expiry.
    If dte is not provided, compute from today's date.
    """
    underlying_price = get_underlying_price(ticker)

    if dte is None:
        today = datetime.today().date()
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte = (exp_date - today).days

    chain = get_calls_for_expiry(ticker, expiry)
    return construct_best_spread_for_chain(
        ticker=ticker,
        expiry=expiry,
        chain=chain,
        underlying_price=underlying_price,
        dte=dte,
    )

def pick_closest_expiry_by_dte(ticker: str, target_dte: int = 60) -> str:
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise ValueError(f"No expiries returned for {ticker}")

    today = datetime.today().date()
    best_exp = None
    best_diff = 10**9

    for exp in expiries:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        diff = abs(dte - target_dte)
        if diff < best_diff:
            best_diff = diff
            best_exp = exp

    if best_exp is None:
        raise ValueError(f"No future expiries for {ticker}")
    return best_exp

if __name__ == "__main__":
    try:
        expiry = pick_closest_expiry_by_dte("AAPL", target_dte=60)
        s = build_spread_for_ticker_and_expiry("AAPL", expiry)
        print("Chosen expiry:", expiry)
        print(s)
    except Exception as e:
        print("Error:", e)
