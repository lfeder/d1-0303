"""Analyze how often wide spreads mean-revert.

For each coin x pair, detects transitions from narrow to wide (>threshold),
then checks if the spread converges back before end of data.
"""

import numpy as np
import pandas as pd
import time
from itertools import combinations

PARQUET_PATH = r"raw\impact_and_funding_0309.parquet"

FEES_BP = {
    "Aster": 4.0, "Binance": 2.0, "Bybit": 2.0,
    "Hyperliquid": 3.5, "Lighter": 0.0, "OKX": 2.0,
}

ABBREV = {
    "Aster": "AS", "Binance": "BN", "Bybit": "BY",
    "Hyperliquid": "HL", "Lighter": "LT", "OKX": "OK",
}


def load_basis():
    print(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH, columns=["exchange", "coin", "minute_utc", "buy_10k", "sell_10k"])
    df["minute_utc"] = pd.to_datetime(df["minute_utc"], utc=True)
    df["epoch_min"] = (df["minute_utc"].astype(np.int64) // 60_000_000_000).astype(np.int64)

    exchanges = sorted(df["exchange"].unique())
    coins = sorted(df["coin"].unique())
    n_ex = len(exchanges)
    n_coins = len(coins)
    ex_to_idx = {e: i for i, e in enumerate(exchanges)}
    coin_to_idx = {c: i for i, c in enumerate(coins)}
    unique_mins = np.sort(df["epoch_min"].unique())
    n_ticks = len(unique_mins)
    min_to_tick = {m: i for i, m in enumerate(unique_mins)}

    tick_idx = np.array([min_to_tick[m] for m in df["epoch_min"].values], dtype=np.int32)
    coin_idx = np.array([coin_to_idx[c] for c in df["coin"].values], dtype=np.int32)
    ex_idx = np.array([ex_to_idx[e] for e in df["exchange"].values], dtype=np.int32)

    buy_tensor = np.full((n_ticks, n_coins, n_ex), np.nan)
    sell_tensor = np.full((n_ticks, n_coins, n_ex), np.nan)
    buy_tensor[tick_idx, coin_idx, ex_idx] = df["buy_10k"].values
    sell_tensor[tick_idx, coin_idx, ex_idx] = df["sell_10k"].values

    # Build pair list (directional)
    pair_indices = []
    pair_names = []
    for a, b in combinations(range(n_ex), 2):
        pair_indices.append((a, b))
        pair_names.append((exchanges[a], exchanges[b]))
        pair_indices.append((b, a))
        pair_names.append((exchanges[b], exchanges[a]))

    fees_arr = np.array([FEES_BP.get(e, 0) for e in exchanges])
    pair_fee_ow = np.array([fees_arr[l] + fees_arr[s] for l, s in pair_indices])

    # Compute entry_basis_bp for all (tick, coin, pair)
    # entry_basis = (sell_s - buy_l) / mid * 10000 - fees
    print("Computing basis for all ticks ...")
    t0 = time.perf_counter()
    basis = np.full((n_ticks, n_coins, len(pair_indices)), np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        for p, (l_idx, s_idx) in enumerate(pair_indices):
            buy_l = buy_tensor[:, :, l_idx]
            sell_s = sell_tensor[:, :, s_idx]
            valid = (buy_l > 0) & (sell_s > 0)
            mid = (buy_l + sell_s) * 0.5
            b = (sell_s - buy_l) / mid * 10000 - pair_fee_ow[p]
            basis[:, :, p] = np.where(valid & (mid > 0), b, np.nan)

    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    return basis, coins, pair_names, pair_fee_ow, n_ticks, unique_mins


def analyze_convergence(basis, coins, pair_names, pair_fee_ow, n_ticks, unique_mins,
                        wide_threshold=50, narrow_threshold=10):
    """
    Find events where basis goes from narrow (<narrow_threshold) to wide (>wide_threshold),
    then track if it converges back below narrow_threshold.
    """
    n_coins = len(coins)
    n_pairs = len(pair_names)

    events = []

    for c in range(n_coins):
        for p in range(n_pairs):
            series = basis[:, c, p]
            # State machine: start in "narrow" only if first valid reading is narrow
            in_narrow = None  # unknown until first valid reading
            wide_tick = None

            for t in range(n_ticks):
                v = series[t]
                if np.isnan(v):
                    continue

                if in_narrow is None:
                    # First valid reading
                    in_narrow = (v < narrow_threshold)
                    continue

                if in_narrow:
                    if v > wide_threshold:
                        # Transition: narrow -> wide
                        in_narrow = False
                        wide_tick = int(t)
                else:
                    # Currently wide - check if converged
                    if v < narrow_threshold:
                        if wide_tick is None:
                            # Started wide, now narrow — not a valid event
                            in_narrow = True
                            continue
                        # Converged!
                        duration = unique_mins[t].item() - unique_mins[wide_tick].item()
                        peak_bp = float(np.nanmax(series[wide_tick:t+1]))
                        events.append({
                            "coin": coins[c],
                            "pair": f"{ABBREV[pair_names[p][0]]}/{ABBREV[pair_names[p][1]]}",
                            "wide_bp": round(float(series[wide_tick]), 1),
                            "peak_bp": round(peak_bp, 1),
                            "duration_min": duration,
                            "converged": True,
                        })
                        in_narrow = True
                        wide_tick = None

            # End of data: if still wide, record as non-converged
            if not in_narrow and wide_tick is not None:
                duration = int(unique_mins[n_ticks-1]) - unique_mins[wide_tick].item()
                peak_bp = float(np.nanmax(series[wide_tick:]))
                events.append({
                    "coin": coins[c],
                    "pair": f"{ABBREV[pair_names[p][0]]}/{ABBREV[pair_names[p][1]]}",
                    "wide_bp": round(float(series[wide_tick]), 1),
                    "peak_bp": round(peak_bp, 1),
                    "duration_min": duration,
                    "converged": False,
                })

    return events


def print_summary(events, wide_threshold):
    if not events:
        print("No events found.")
        return

    total = len(events)
    converged = sum(1 for e in events if e["converged"])
    not_converged = total - converged

    print(f"\n{'='*70}")
    print(f"  SPREAD CONVERGENCE ANALYSIS  (wide = >{wide_threshold} bp, narrow = <10 bp)")
    print(f"{'='*70}")
    print(f"  Total narrow->wide events:  {total}")
    print(f"  Converged back:             {converged} ({converged/total*100:.1f}%)")
    print(f"  Still wide at end of data:  {not_converged} ({not_converged/total*100:.1f}%)")

    conv_events = [e for e in events if e["converged"]]
    if conv_events:
        durs = [e["duration_min"] for e in conv_events]
        print(f"\n  Converged events:")
        print(f"    Median time to converge:  {int(np.median(durs))}m ({int(np.median(durs))//60}h {int(np.median(durs))%60}m)")
        print(f"    Mean time to converge:    {int(np.mean(durs))}m ({int(np.mean(durs))//60}h {int(np.mean(durs))%60}m)")
        print(f"    Max time to converge:     {max(durs)}m ({max(durs)//60}h {max(durs)%60}m)")
        pcts = [25, 50, 75, 90, 95]
        vals = np.percentile(durs, pcts)
        print(f"    Percentiles (min):        " + "  ".join(f"p{p}={int(v)}" for p, v in zip(pcts, vals)))

    # Per-pair summary
    pair_stats = {}
    for e in events:
        pair = e["pair"]
        if pair not in pair_stats:
            pair_stats[pair] = {"total": 0, "converged": 0, "durs": []}
        pair_stats[pair]["total"] += 1
        if e["converged"]:
            pair_stats[pair]["converged"] += 1
            pair_stats[pair]["durs"].append(e["duration_min"])

    print(f"\n  {'Pair':<10} {'Events':>7} {'Conv':>6} {'Rate':>7} {'Med(m)':>8} {'Mean(m)':>9}")
    print("  " + "-" * 50)
    for pair in sorted(pair_stats.keys(), key=lambda p: -pair_stats[p]["total"]):
        s = pair_stats[pair]
        rate = s["converged"] / s["total"] * 100 if s["total"] else 0
        med = int(np.median(s["durs"])) if s["durs"] else 0
        mean = int(np.mean(s["durs"])) if s["durs"] else 0
        print(f"  {pair:<10} {s['total']:>7} {s['converged']:>6} {rate:>6.1f}% {med:>8} {mean:>9}")

    # Non-converged details
    if not_converged > 0:
        print(f"\n  Non-converged events ({not_converged}):")
        nc = [e for e in events if not e["converged"]]
        nc.sort(key=lambda e: -e["peak_bp"])
        for e in nc[:20]:
            print(f"    {e['coin']:<8} {e['pair']:<8} wide={e['wide_bp']:>6.1f}bp  peak={e['peak_bp']:>6.1f}bp  open for {e['duration_min']}m")
        if len(nc) > 20:
            print(f"    ... and {len(nc)-20} more")

    print(f"{'='*70}")


if __name__ == "__main__":
    basis, coins, pair_names, pair_fee_ow, n_ticks, unique_mins = load_basis()

    for thresh in [50, 100]:
        events = analyze_convergence(basis, coins, pair_names, pair_fee_ow, n_ticks, unique_mins,
                                     wide_threshold=thresh, narrow_threshold=10)
        print_summary(events, thresh)
