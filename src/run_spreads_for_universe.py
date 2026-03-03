import pandas as pd
from datetime import datetime
import yfinance as yf
from dataclasses import asdict

from spread_constructor import (
    build_spread_for_ticker_and_expiry, 
    BullCallSpread,
)

ticker_source_csv = "options_liquidity_filtered.csv"

TARGET_DTE = 60
MIN_DTE = 25    # hard floor – no 2–9 DTE stuff
MAX_DTE = 80    # upper bound – avoid super-long stuff for this strategy

output_csv = "constructed_spreads_60dte.csv"


def load_universe(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    # adjust column name if needed (e.g. "Ticker")
    tickers = df["ticker"].dropna().unique()
    return list(tickers)


def get_sorted_expiries_with_dte(ticker: str, target_dte: int = TARGET_DTE): 
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries: 
        return []
    
    today = datetime.today().date()
    exp_dte_pairs = []

    for exp_str in expiries: 
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue

        # Only keep expiries within [MIN_DTE, MAX_DTE]
        if dte < MIN_DTE or dte > MAX_DTE:
            continue

        exp_dte_pairs.append((exp_str, dte))

    # Sort by |DTE - target|
    exp_dte_pairs.sort(key=lambda x: abs(x[1] - target_dte))
    return exp_dte_pairs


def find_best_spread_for_ticker(ticker: str) -> tuple[BullCallSpread | None, str | None, int | None]:
    exp_dte_pairs = get_sorted_expiries_with_dte(ticker, TARGET_DTE)
    if not exp_dte_pairs:
        return None, None, None
    
    for expiry, dte in exp_dte_pairs:
        try: 
            spread = build_spread_for_ticker_and_expiry(ticker, expiry, dte=dte)
            if spread is not None: 
                return spread, expiry, dte
        except Exception as e: 
            print(f"Error trying {ticker} @ {expiry}: {e}")
            continue

    return None, None, None


def main():
    tickers = load_universe(ticker_source_csv)
    print(f"Loaded {len(tickers)} tickers from {ticker_source_csv}")

    all_spreads = []

    for t in tickers: 
        print(f"\n=== {t} ===")
        spread, expiry, dte = find_best_spread_for_ticker(t)

        if spread is None: 
            print("No valid 25–80 DTE spread found under current constraints")
            continue

        print(f"Chosen expiry: {expiry} (DTE ≈ {dte})")
        print(f"Spread: {spread}")

        spread_dict = asdict(spread)
        spread_dict["dte"] = dte
        all_spreads.append(spread_dict)
    
    if not all_spreads: 
        print("\nNo spreads found for any ticker.")
        return
    
    result_df = pd.DataFrame(all_spreads)
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(result_df)} constructed spreads to '{output_csv}'.")


if __name__ == "__main__":
    main()
