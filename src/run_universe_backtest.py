#========================
# run_universe_backtest.py
#========================
import pandas as pd
import numpy as np
from pathlib import Path

from single_ticker_synthetic_backtest import run_synthetic_backtest_roll

# ---- CONFIG ----
UNIVERSE_CSV = "options_liquidity_filtered.csv"   # your updated list
TICKER_COL   = "ticker"                           # change if needed
START_DATE   = "2021-01-01"
END_DATE     = "2025-12-31"
TARGET_DTE   = 60

OUT_DIR = Path("backtest_outputs")
OUT_DIR.mkdir(exist_ok=True)

ALL_TRADES_CSV = OUT_DIR / "synthetic_trades_ALL.csv"
SUMMARY_CSV    = OUT_DIR / "synthetic_summary_by_ticker.csv"
FAILURES_CSV   = OUT_DIR / "synthetic_failures.csv"
PORTFOLIO_CSV  = OUT_DIR / "portfolio_equity_curve.csv"


def load_universe(csv_path: str, ticker_col: str = "ticker") -> list[str]:
    df = pd.read_csv(csv_path)
    if ticker_col not in df.columns:
        raise KeyError(f"Ticker column '{ticker_col}' not found. Columns: {list(df.columns)}")

    tickers = (
        df[ticker_col]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .unique()
        .tolist()
    )
    return tickers


def summarize_trades(trades: pd.DataFrame) -> dict:
    """
    Per-ticker summary metrics for your report.
    Requires columns: pnl, ret_on_debit
    """
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_ret_on_debit": np.nan,
            "median_ret_on_debit": np.nan,
            "avg_pnl": np.nan,
            "median_pnl": np.nan,
            "max_drawdown_ret": np.nan,
            "cum_ret_simple": np.nan,
        }

    t = trades.copy()

    # win rate uses pnl > 0 on ALL rows
    win_rate = float(t["pnl"].gt(0).mean())

    # clean returns
    t["ret_on_debit"] = pd.to_numeric(t["ret_on_debit"], errors="coerce")
    t_valid = t.dropna(subset=["ret_on_debit"]).copy()

    avg_pnl = float(t["pnl"].mean())
    median_pnl = float(t["pnl"].median())

    # if returns all NaN, still return pnl + win stats
    if t_valid.empty:
        return {
            "trades": int(len(t)),
            "win_rate": win_rate,
            "avg_ret_on_debit": np.nan,
            "median_ret_on_debit": np.nan,
            "avg_pnl": avg_pnl,
            "median_pnl": median_pnl,
            "max_drawdown_ret": np.nan,
            "cum_ret_simple": np.nan,
        }

    # equity curve per ticker (order matters; entry_date is safest)
    if "entry_date" in t_valid.columns:
        t_valid["entry_date"] = pd.to_datetime(t_valid["entry_date"], errors="coerce")
        t_valid = t_valid.sort_values("entry_date")

    eq = (1.0 + t_valid["ret_on_debit"]).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0

    return {
        "trades": int(len(t)),
        "win_rate": win_rate,
        "avg_ret_on_debit": float(t_valid["ret_on_debit"].mean()),
        "median_ret_on_debit": float(t_valid["ret_on_debit"].median()),
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "max_drawdown_ret": float(dd.min()),
        "cum_ret_simple": float(eq.iloc[-1] - 1.0),
    }


def main():
    tickers = load_universe(UNIVERSE_CSV, TICKER_COL)
    print(f"Loaded {len(tickers)} tickers from {UNIVERSE_CSV}")

    all_trades_list: list[pd.DataFrame] = []
    summaries: list[dict] = []
    failures: list[dict] = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"\n[{i}/{len(tickers)}] Backtesting {ticker} ...")

        try:
            trades = run_synthetic_backtest_roll(
                ticker=ticker,
                start=START_DATE,
                end=END_DATE,
                target_dte=TARGET_DTE,
                step_after_expiry_days=1,
            )

            if trades is None or trades.empty:
                failures.append({
                    "ticker": ticker,
                    "type": "NO_TRADES",
                    "message": "No trades constructed under constraints",
                })
                continue

            all_trades_list.append(trades)

            s = summarize_trades(trades)
            s["ticker"] = ticker
            summaries.append(s)

            print(f"  -> Trades: {len(trades)} | Win rate: {s['win_rate']:.2%} | CumRet: {s['cum_ret_simple']:.2%}")

        except Exception as e:
            failures.append({
                "ticker": ticker,
                "type": "ERROR",
                "message": str(e),
            })
            print(f"  -> ERROR on {ticker}: {e}")

    # ---- Save summary + failures ALWAYS ----
    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["cum_ret_simple", "win_rate"], ascending=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary -> {SUMMARY_CSV} ({len(summary_df)} tickers)")

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(FAILURES_CSV, index=False)
    print(f"Saved failures log -> {FAILURES_CSV} ({len(failures_df)} rows)")

    # ---- Save ALL trades if any ----
    if all_trades_list:
        all_trades_df = pd.concat(all_trades_list, ignore_index=True)
        all_trades_df.to_csv(ALL_TRADES_CSV, index=False)
        print(f"Saved ALL trades -> {ALL_TRADES_CSV} ({len(all_trades_df)} rows)")
    else:
        print("\nNo trades across entire universe.")
        return  # nothing else to compute

    # Sort by time so compounding is chronological (across all tickers)
    if "entry_date" in all_trades_df.columns:
        all_trades_df["entry_date"] = pd.to_datetime(all_trades_df["entry_date"], errors="coerce")
        all_trades_df = all_trades_df.sort_values("entry_date")

    returns = pd.to_numeric(all_trades_df["ret_on_debit"], errors="coerce").dropna()

    # ---- Quick overall distribution stats ----
    win_pct = float((all_trades_df["pnl"] > 0).mean())

    print("\nOverall return-on-debit distribution:")
    print(f"Count: {int(len(returns))}")
    print(f"Mean:  {returns.mean():.4f}")
    print(f"Median:{returns.median():.4f}")
    print(f"Win%:  {win_pct:.2%}")


if __name__ == "__main__":
    main()
