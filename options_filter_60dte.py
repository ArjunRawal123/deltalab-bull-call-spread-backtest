#========================
# options_filter_60dte.py
#========================
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

filter_1 = "Updated_stock_filtered.csv"

# Option filtering constraints
target_DTE = 60
min_OPEN_INTEREST = 500
min_OPTION_VOLUME = 50
min_OI_OV_ratio = 5
max_OI_OV_ratio = 50
max_BAS = 0.10

def load_tickers(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    tickers = df["Ticker"].dropna().unique()
    return tickers

def choose_expiry_closest_to_target(options_list, target_DTE: int) -> str | None:
    if not options_list:
        return None

    today = datetime.today().date()

    best_expiry = None
    best_distance = None

    for exp_str in options_list:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days

        if dte <= 0:
            continue

        distance = abs(dte - target_DTE)

        if (best_distance is None) or (distance < best_distance):
            best_distance = distance
            best_expiry = exp_str

    return best_expiry   

def filter_options_for_ticker(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    expires = tk.options
    if not expires:
        return pd.DataFrame()

    chosen_expiry = choose_expiry_closest_to_target(expires, target_DTE)
    if chosen_expiry is None:
        return pd.DataFrame()

    chain = tk.option_chain(chosen_expiry).calls
    if chain.empty:
        return pd.DataFrame()

    df = chain.copy()

    df["volume_nonzero"] = df["volume"].replace(0, np.nan)
    df["oi_ov"] = df["openInterest"] / df["volume_nonzero"]
    df["bas"] = df["ask"] - df["bid"]

    mask = (
        (df["openInterest"] >= min_OPEN_INTEREST) &
        (df["volume"] >= min_OPTION_VOLUME) &
        (df["oi_ov"] >= min_OI_OV_ratio) & (df["oi_ov"] <= max_OI_OV_ratio) &
        (df["bas"] <= max_BAS)
    )

    filtered = df[mask].copy()

    if not filtered.empty:
        filtered["ticker"] = ticker
        filtered["expiry"] = chosen_expiry

    return filtered

def main():
    print(">>> Script started")

    tickers = load_tickers(filter_1)
    print(f"Loaded {len(tickers)} tickers from CSV.")

    all_contracts = []

    for t in tickers:
        try:
            print(f"Processing {t}...")
            passed = filter_options_for_ticker(t)

            if not passed.empty:
                all_contracts.append(passed)

        except Exception as e:
            print(f"Error for {t}: {e}")

    if not all_contracts:
        print("No contracts met the criteria.")
        return

    result_df = pd.concat(all_contracts, ignore_index=True)

    final_ticker_pool = sorted(result_df["ticker"].unique())
    print(f"\nNumber of tickers that passed: {len(final_ticker_pool)}")
    print("Tickers:", final_ticker_pool)

    result_df.to_csv("options_liquidity_filtered.csv", index=False)
    print("\nSaved detailed contracts to 'options_liquidity_filtered.csv'.")

if __name__ == "__main__":
    main()
