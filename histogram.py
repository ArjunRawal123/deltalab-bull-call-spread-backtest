import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_return_hist(
    rets, out_path, *,
    bin_width=0.05,
    label_threshold=25,
    logy=False,
    title="Per-Trade Return Distribution"
):
    rets = np.asarray(rets)
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        raise ValueError("No valid returns to plot.")

    rets = np.clip(rets, -1.0, None)

    bins = np.arange(-1.0, rets.max() + bin_width, bin_width)

    counts, edges = np.histogram(rets, bins=bins)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2

    full_loss_rate = float(np.mean(np.isclose(rets, -1.0)))

    plt.figure(figsize=(11, 6))
    plt.bar(centers, counts, width=widths * 0.95, align="center", edgecolor="black")

    if logy:
        plt.yscale("log")

    plt.grid(True, axis="y", alpha=0.25)

    plt.text(
        0.02, 0.95,
        f"Trades: {len(rets)}\nFull loss (-100%): {full_loss_rate:.1%}",
        transform=plt.gca().transAxes,
        va="top"
    )

    plt.title(title)
    plt.xlabel("Return (P/L ÷ Debit)")
    plt.ylabel("Number of Trades")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved -> {out_path}")

def main():
    df = pd.read_csv("backtest_outputs/synthetic_trades_ALL.csv")
    rets = pd.to_numeric(df["ret_on_debit"], errors="coerce").dropna().to_numpy()

    plot_return_hist(
        rets,
        "backtest_outputs/hist_per_trade_returns_logy.png",
        bin_width=0.05,
        label_threshold=20,
        logy=True,
        title="Per-Trade Return Distribution (Log Y)"
    )

if __name__ == "__main__":
    main()
