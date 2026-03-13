"""Compute max absolute price moves per coin across time windows.
Outputs margin/moves.json for the margin simulator's third tab.
"""
import pandas as pd
import numpy as np
import json

PARQUET = "raw/impact_and_funding_0309.parquet"
OUT = "margin/moves.json"

WINDOWS = [1, 10, 30, 60, 120, 240, 480, 1440, 7200]  # minutes
WIN_LABELS = ["1m", "10m", "30m", "1h", "2h", "4h", "8h", "24h", "120h"]

def main():
    df = pd.read_parquet(PARQUET, columns=["exchange", "coin", "minute_utc", "buy_10k", "sell_10k"])
    df["mid"] = (df["buy_10k"] + df["sell_10k"]) * 0.5
    df = df.dropna(subset=["mid"])
    df = df.sort_values(["coin", "exchange", "minute_utc"])

    # For each coin, use all exchanges' data to find the worst move
    # across any exchange for that coin in each window
    coins = sorted(df["coin"].unique())
    result = {}

    for coin in coins:
        dc = df[df["coin"] == coin]
        max_moves = {}

        for w, label in zip(WINDOWS, WIN_LABELS):
            worst = 0.0
            for exch, dce in dc.groupby("exchange"):
                prices = dce["mid"].values
                if len(prices) <= w:
                    continue
                # Rolling max abs % move over window of w minutes
                # Compare price[i] to price[i+w]
                p_start = prices[:-w]
                p_end = prices[w:]
                moves = np.abs(p_end - p_start) / p_start * 100
                if len(moves) > 0:
                    worst = max(worst, float(np.max(moves)))
            max_moves[label] = round(worst, 2)

        result[coin] = max_moves
        print(f"{coin}: {max_moves}")

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT} ({len(result)} coins)")

if __name__ == "__main__":
    main()
