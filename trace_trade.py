import numpy as np
import pandas as pd
from datetime import datetime, timezone

df = pd.read_parquet('raw/impact_and_funding_0309.parquet', columns=['exchange','coin','minute_utc','buy_10k','sell_10k'])
df['minute_utc'] = pd.to_datetime(df['minute_utc'], utc=True)
df['epoch_min'] = (df['minute_utc'].astype(np.int64) // 60_000_000_000).astype(np.int64)

coin = 'ZRO'
l_ex, s_ex = 'Binance', 'Lighter'
fee = 2  # BN 2 + LT 0

sub = df[(df['coin']==coin) & (df['exchange'].isin([l_ex, s_ex]))]
sub = sub.pivot_table(index='epoch_min', columns='exchange', values=['buy_10k','sell_10k']).sort_index()

entry_epoch = int(pd.Timestamp('2026-02-11 00:46', tz='UTC').timestamp()) // 60
exit_epoch = int(pd.Timestamp('2026-02-11 01:49', tz='UTC').timestamp()) // 60

hdr = f"{'Tick':<18} {'BN_bid':>8} {'BN_ask':>8} {'BN_spd':>7} {'LT_bid':>8} {'LT_ask':>8} {'LT_spd':>7} {'basis':>8}"
print(f'ZRO  L=BN, S=LT  (fee={fee}bp)')
print(f'basis = (LT_bid - BN_ask) / mid * 10000 - {fee}')
print()
print(hdr)
print('-' * len(hdr))

for em in sorted(sub.index):
    if em < entry_epoch - 3 or em > exit_epoch + 2:
        continue
    if em > entry_epoch + 5 and em < exit_epoch - 2:
        continue

    bn_bid = sub.loc[em, ('sell_10k', l_ex)]
    bn_ask = sub.loc[em, ('buy_10k', l_ex)]
    lt_bid = sub.loc[em, ('sell_10k', s_ex)]
    lt_ask = sub.loc[em, ('buy_10k', s_ex)]

    if pd.isna(bn_bid) or pd.isna(lt_bid):
        continue

    mid = (bn_ask + lt_bid) / 2
    basis = (lt_bid - bn_ask) / mid * 10000 - fee

    ts = datetime.fromtimestamp(em * 60, tz=timezone.utc).strftime('%m-%d %H:%M')
    marker = ''
    if em == entry_epoch: marker = '  <-- ENTRY'
    if em == exit_epoch: marker = '  <-- EXIT'

    print(f'{ts:<18} {bn_bid:>8.4f} {bn_ask:>8.4f} {bn_ask-bn_bid:>7.4f} {lt_bid:>8.4f} {lt_ask:>8.4f} {lt_ask-lt_bid:>7.4f} {basis:>+8.1f}{marker}')

# Manual P&L
entry_buy_l = float(sub.loc[entry_epoch, ('buy_10k', l_ex)])
entry_sell_s = float(sub.loc[entry_epoch, ('sell_10k', s_ex)])
exit_sell_l = float(sub.loc[exit_epoch, ('sell_10k', l_ex)])
exit_buy_s = float(sub.loc[exit_epoch, ('buy_10k', s_ex)])

mid = (entry_buy_l + entry_sell_s) / 2
qty = 10000 / mid

pl_long = (exit_sell_l - entry_buy_l) * qty
pl_short = (entry_sell_s - exit_buy_s) * qty
fees_usd = fee / 10000 * 10000 * 2

print()
print(f'Entry: buy BN @ {entry_buy_l:.4f}, sell LT @ {entry_sell_s:.4f}')
print(f'  spread captured = {entry_sell_s - entry_buy_l:.4f}')
print(f'Exit:  sell BN @ {exit_sell_l:.4f}, buy LT @ {exit_buy_s:.4f}')
print(f'  spread paid = {exit_buy_s - exit_sell_l:.4f}')
print(f'qty = {qty:.2f}')
print()
print(f'Long leg:  ({exit_sell_l:.4f} - {entry_buy_l:.4f}) * {qty:.2f} = ${pl_long:+.2f}')
print(f'Short leg: ({entry_sell_s:.4f} - {exit_buy_s:.4f}) * {qty:.2f} = ${pl_short:+.2f}')
print(f'Fees: ${fees_usd:.2f}')
print(f'Net: ${pl_long + pl_short - fees_usd:+.2f}')
