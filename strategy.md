## Entry
For every minute, score all `coin x exchange-pair` combinations.

Long on cheap exchange (l_ex), short on expensive exchange (s_ex).
- `buy_l` = l_ex ask (price to go long)
- `sell_s` = s_ex bid (price to go short)
- `buy_s` = s_ex ask (price to unwind short)
- `sell_l` = l_ex bid (price to unwind long)

```
mid            = (buy_l + sell_s) / 2
qty            = position_size / mid
entry_basis_bp = (sell_s - buy_l) / mid x 10000 - (fee_l_bp + fee_s_bp)
exit_cost_bp   = ((buy_l - sell_l) + (buy_s - sell_s)) / mid x 10000 + (fee_l_bp + fee_s_bp)
entry_score_bp = entry_basis_bp - exit_cost_bp
```

**Example entry** — ETH on Lighter (0 bp fee) and Bybit (2 bp fee). Long BY (l_ex), short LT (s_ex).
```
LT (s_ex): bid=10095, ask=10105    -->  sell_s=10095, buy_s=10105
BY (l_ex): bid=9990,  ask=10010    -->  sell_l=9990,  buy_l=10010

mid            = (10010 + 10095) / 2 = 10052.5
qty            = 10000 / 10052.5 = 0.99478 coins
entry_basis_bp = (10095 - 10010) / 10052.5 x 10000 - 2 = 84.6 - 2 = 82.6 bp
exit_cost_bp   = ((10010 - 9990) + (10105 - 10095)) / 10052.5 x 10000 + 2
               = (20 + 10) / 10052.5 x 10000 + 2 = 29.8 + 2 = 31.8 bp
entry_score_bp = 82.6 - 31.8 = 50.8 bp  --> ENTRY
```

Enter the **single best** scoring opportunity if `entry_score_bp > 10` and:
- Fewer than 15 active positions
- Not already holding that coin
- Execution at **T+1** (signal at T, fill at next minute's prices)

## Exit
Each minute, score active positions for exit using current prices:

```
mid           = (buy_s + sell_l) / 2
exit_basis_bp = (buy_s - sell_l) / mid x 10000 + (fee_l_bp + fee_s_bp)
exit_score    = entry_basis_bp - exit_basis_bp
```

Two exit triggers (checked in order):
1. **Signal**: `exit_score > 10 bp`
2. **Timeout**: held >= 480 minutes (8 hours)

**Example exit** (T=5, basis converged, spreads unchanged):
```
LT (s_ex): bid=10050, ask=10060    -->  sell_s=10050, buy_s=10060
BY (l_ex): bid=10035, ask=10045    -->  sell_l=10035, buy_l=10045

mid           = (10060 + 10035) / 2 = 10047.5
exit_basis_bp = (10060 - 10035) / 10047.5 x 10000 + 2 = 24.9 + 2 = 26.9 bp
```

Execution at **T+1** (same as entry).

## P&L
```
pl_bp  = entry_basis_bp - exit_basis_bp
pl_usd = pl_bp / 10000 x position_size
```

Using the example above:
```
pl_bp  = 82.6 - 26.9 = 55.7 bp
pl_usd = 55.7 / 10000 x 10000 = $55.70
```

## Parameters
| Parameter | Default |
|-----------|---------|
| Position size | $10,000 per side |
| Max positions | 15 |
| Entry threshold | 10 bp |
| Exit threshold | 10 bp |
| Max hold time | 480 min (8h) |
| Capital (for annualized calc) | $150,000 |
