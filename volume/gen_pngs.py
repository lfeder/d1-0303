"""Generate rank and exchange PNG visualizations with per-day ranking."""
import json, os
from PIL import Image, ImageDraw, ImageFont

with open(os.path.join(os.path.dirname(__file__), "volume.json")) as f:
    DATA = json.load(f)

dates = DATA["dates"]
coins = DATA["coins"]
exchDaily = DATA["exchDaily"]
exchanges = DATA["exchanges"]
n = len(dates)

EXCH_ABBR = {"binance": "BN", "bybit": "BB", "okx": "OKX", "hyperliquid": "HL", "aster": "AS", "lighter": "LT"}

# Precompute per-day ranked vols
daily_ranks = {}
for coin in coins:
    exkeys = list(coin["e"].keys())
    nex = len(exkeys)
    ranked = [[] for _ in range(nex)]
    for d in range(n):
        day_vols = sorted([coin["e"][e][d] or 0 for e in exkeys], reverse=True)
        for r in range(nex):
            ranked[r].append(day_vols[r])
    daily_ranks[coin["s"]] = ranked


def build_rank_panel(target_rank):
    ridx = target_rank - 1
    rows = []
    for coin in coins:
        dr = daily_ranks[coin["s"]]
        if ridx >= len(dr):
            continue
        vols = dr[ridx]
        avg = sum(vols) / n
        rows.append({"symbol": coin["s"], "vols": vols, "avg": avg})
    rows.sort(key=lambda x: -x["avg"])
    summary = []
    for rank in range(1, 7):
        rv = [0] * n
        for coin in coins:
            dr = daily_ranks[coin["s"]]
            if rank > len(dr):
                continue
            for i in range(n):
                rv[i] += dr[rank - 1][i]
        summary.append(("#%d" % rank, rv, sum(rv) / n))
    return rows, summary


def build_exch_panel(target_ex):
    rows = []
    for coin in coins:
        if target_ex not in coin["e"]:
            continue
        vols = coin["e"][target_ex]
        avg = sum(vols) / n
        rows.append({"symbol": coin["s"], "vols": vols, "avg": avg})
    rows.sort(key=lambda x: -x["avg"])
    sorted_ex = sorted(exchanges, key=lambda ex: -sum(exchDaily[ex]))
    summary = []
    for ex in sorted_ex:
        vols = exchDaily[ex]
        summary.append((EXCH_ABBR.get(ex, ex), vols, sum(vols) / n))
    return rows, summary


def vol_rgb(v):
    if not v:
        return (238, 238, 232)
    if v >= 1e8:
        return (26, 122, 26)
    if v >= 1e7:
        return (124, 196, 124)
    if v >= 1e6:
        return (220, 200, 80)
    return (200, 100, 100)


def sum_rgb(v):
    if not v:
        return (238, 238, 232)
    if v >= 25e9:
        return (26, 122, 26)
    if v >= 5e9:
        return (124, 196, 124)
    if v >= 1e9:
        return (220, 200, 80)
    return (200, 100, 100)


def fmt(v):
    if not v:
        return ""
    if v >= 1e9:
        return "%.1fB" % (v / 1e9)
    if v >= 1e6:
        return "%.1fM" % (v / 1e6)
    if v >= 1e3:
        return "%dK" % (v / 1e3)
    return str(int(v))


font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 9)
font_bold = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 9)
font_title = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 11)
font_cutoff = ImageFont.truetype("C:/Windows/Fonts/impact.ttf", 144)

ROW_H = 9
HDR_H = 14
SEP_H = 10
TITLE_H = 16
LABEL_H = 160
LINE_THICK = 20


def render_panels(panels, out_filename):
    COL_NUM = 22
    COL_SYM = 58
    COL_AVG = 42
    LEFT = COL_NUM + COL_SYM + COL_AVG
    BAR_W = 280
    PANEL_W = LEFT + BAR_W + 2
    GAP = 6

    built = {}
    max_rows = 0
    for title, builder in panels:
        rows, summary = builder()
        built[title] = (rows, summary)
        max_rows = max(max_rows, len(rows))

    np = len(panels)
    W = np * PANEL_W + (np - 1) * GAP + 8
    n_summary = len(built[panels[0][0]][1])
    H = TITLE_H + HDR_H + n_summary * ROW_H + SEP_H + max_rows * ROW_H + LABEL_H + LINE_THICK + 4

    img = Image.new("RGB", (W, H), (245, 245, 240))
    draw = ImageDraw.Draw(img)

    for pi, (title, _) in enumerate(panels):
        rows, summary = built[title]
        x_off = 4 + pi * (PANEL_W + GAP)

        cutoff_idx = len(rows)
        for ci, r in enumerate(rows):
            if r["avg"] < 1e6:
                cutoff_idx = ci
                break

        draw.text((x_off + 2, 3), title, fill=(34, 34, 34), font=font_title)
        y = TITLE_H

        # Header
        draw.rectangle([x_off, y, x_off + PANEL_W, y + HDR_H], fill=(232, 232, 226))
        draw.text((x_off + 2, y + 2), "#", fill=(68, 68, 68), font=font_bold)
        draw.text((x_off + COL_NUM + 1, y + 2), "Coin", fill=(68, 68, 68), font=font_bold)
        draw.text((x_off + COL_NUM + COL_SYM + 1, y + 2), "Avg", fill=(68, 68, 68), font=font_bold)
        tick_step = max(1, n // 8)
        for i in range(0, n, tick_step):
            x = int(x_off + LEFT + (i / n) * BAR_W)
            draw.line([(x, y), (x, y + HDR_H)], fill=(204, 204, 204))
            draw.text((x + 1, y + 2), dates[i][5:], fill=(136, 136, 136), font=font)
        y += HDR_H

        # Summary
        for slabel, svols, savg in summary:
            draw.rectangle([x_off, y, x_off + LEFT, y + ROW_H], fill=(224, 224, 216))
            draw.text((x_off + COL_NUM + 1, y), slabel, fill=(34, 34, 34), font=font_bold)
            draw.text((x_off + COL_NUM + COL_SYM + 1, y), fmt(savg), fill=(34, 34, 34), font=font)
            pw = BAR_W / n
            for i in range(n):
                rgb = sum_rgb(svols[i])
                x0 = int(x_off + LEFT + i * pw)
                x1 = int(x_off + LEFT + (i + 1) * pw)
                draw.rectangle([x0, y, x1, y + ROW_H - 2], fill=rgb)
            y += ROW_H

        # Separator
        draw.rectangle([x_off, y, x_off + PANEL_W, y + SEP_H], fill=(216, 220, 228))
        draw.text((x_off + COL_NUM + 1, y), "Coins", fill=(34, 34, 34), font=font_bold)
        y += SEP_H

        # Coin rows
        for i, r in enumerate(rows):
            if i == cutoff_idx:
                label_text = str(cutoff_idx)
                bbox = font_cutoff.getbbox(label_text)
                tw = bbox[2] - bbox[0]
                lx = x_off + (PANEL_W - tw) // 2
                ly = y + 4
                draw.text((lx, ly - bbox[1]), label_text, fill=(0, 0, 0), font=font_cutoff)
                y += LABEL_H
                draw.rectangle([x_off, y, x_off + PANEL_W, y + LINE_THICK], fill=(0, 0, 0))
                y += LINE_THICK

            bg = (240, 240, 235) if i % 2 else (250, 250, 245)
            draw.rectangle([x_off, y, x_off + LEFT, y + ROW_H], fill=bg)
            draw.text((x_off + 2, y), str(i + 1), fill=(153, 153, 153), font=font)
            draw.text((x_off + COL_NUM + 1, y), r["symbol"], fill=(34, 34, 34), font=font_bold)
            draw.text((x_off + COL_NUM + COL_SYM + 1, y), fmt(r["avg"]), fill=(68, 68, 68), font=font)
            pw = BAR_W / n
            for i2 in range(n):
                rgb = vol_rgb(r["vols"][i2])
                x0 = int(x_off + LEFT + i2 * pw)
                x1 = int(x_off + LEFT + (i2 + 1) * pw)
                draw.rectangle([x0, y, x1, y + ROW_H - 2], fill=rgb)
            y += ROW_H

    img.save(out_filename)
    sz = os.path.getsize(out_filename) / 1e6
    print("Saved %s -- %dx%dpx (%.1fMB)" % (out_filename, W, H, sz))
    for title, _ in panels:
        rows = built[title][0]
        cutoff = sum(1 for r in rows if r["avg"] >= 1e6)
        print("  %s: %d coins >= $1M/day" % (title, cutoff))


# Rank PNG
outdir = os.path.dirname(__file__)
rank_panels = [("Rank #%d" % r, lambda r=r: build_rank_panel(r)) for r in [2, 3, 4, 5, 6]]
render_panels(rank_panels, os.path.join(outdir, "volume-rank2-6-all.png"))

print()

# Exchange PNG (sorted by total vol desc)
sorted_ex = sorted(exchanges, key=lambda ex: -sum(exchDaily[ex]))
exch_panels = [(EXCH_ABBR.get(ex, ex), lambda ex=ex: build_exch_panel(ex)) for ex in sorted_ex]
render_panels(exch_panels, os.path.join(outdir, "volume-by-exchange.png"))
