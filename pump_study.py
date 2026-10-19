#!/usr/bin/env python3
"""
pump_study.py - event study for volume-spike pumps on small Binance coins.

Three steps, run in order:
  python pump_study.py universe --market spot      # pick the coin list
  python pump_study.py fetch    --market spot      # download 1m candles (cached, resumable)
  python pump_study.py analyze  --market spot      # find spikes, measure what happens next

Uses only public Binance endpoints. No API key, no orders, read-only.
Needs: pip install requests pandas numpy
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

MARKETS = {
    "spot": {
        "base": "https://api.binance.com",
        "info": "/api/v3/exchangeInfo",
        "ticker": "/api/v3/ticker/24hr",
        "klines": "/api/v3/klines",
        "limit": 1000,          # weight 2 per call, 6000/min allowed
        "weight_soft": 4500,
        "fee": 0.001,           # 0.1% per side (no BNB discount)
    },
    "futures": {
        "base": "https://fapi.binance.com",
        "info": "/fapi/v1/exchangeInfo",
        "ticker": "/fapi/v1/ticker/24hr",
        "klines": "/fapi/v1/klines",
        "limit": 1000,          # weight 5 per call, 2400/min allowed
        "weight_soft": 1800,
        "fee": 0.0005,          # 0.05% taker per side
    },
}

STABLES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDE", "PYUSD", "USD1",
    "AEUR", "EUR", "EURI", "RLUSD", "BFUSD", "XUSD", "USDS", "PAXG", "WBTC", "WBETH",
}
MAJORS = {"BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "TRX", "LTC", "BCH"}

HORIZONS = [1, 3, 5, 15, 30]          # minutes after entry
GRID_TP = [0.03, 0.05, 0.08, 0.12, 0.20]
GRID_SL = [0.02, 0.03, 0.05]
GRID_HOLD = [5, 15, 30]


# ---------------------------------------------------------------- HTTP

def http_get(session, url, params, mkt):
    for attempt in range(8):
        try:
            r = session.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt)
            print(f"  network error ({e}); retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code in (418, 429):
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  rate limited ({r.status_code}); waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code in (403, 451):
            sys.exit(f"Binance refused the request ({r.status_code}). "
                     "Your location/IP may be blocked for this endpoint.")
        if r.status_code >= 500:
            time.sleep(min(60, 2 ** attempt))
            continue
        r.raise_for_status()
        used = r.headers.get("x-mbx-used-weight-1m")
        if used and int(used) > mkt["weight_soft"]:
            print(f"  weight {used}/min, cooling down 15s")
            time.sleep(15)
        return r.json()
    raise RuntimeError(f"giving up on {url} {params}")


# ---------------------------------------------------------------- universe

def cmd_universe(a):
    mkt = MARKETS[a.market]
    s = requests.Session()
    info = http_get(s, mkt["base"] + mkt["info"], {}, mkt)
    tickers = http_get(s, mkt["base"] + mkt["ticker"], {}, mkt)
    vol24 = {t["symbol"]: float(t["quoteVolume"]) for t in tickers}

    cands = []
    for x in info["symbols"]:
        if x.get("status") != "TRADING" or x.get("quoteAsset") != "USDT":
            continue
        if a.market == "futures" and x.get("contractType") != "PERPETUAL":
            continue
        if a.market == "spot" and not x.get("isSpotTradingAllowed", True):
            continue
        base = x["baseAsset"]
        if base in STABLES or base in MAJORS:
            continue
        v = vol24.get(x["symbol"], 0.0)
        if a.min_vol24 <= v <= a.max_vol24:
            cands.append((x["symbol"], v))

    print(f"{len(cands)} symbols pass the 24h volume filter "
          f"(${a.min_vol24/1e6:.1f}M - ${a.max_vol24/1e6:.0f}M). Checking daily ranges...")

    rows = []
    for sym, v in cands:
        d = http_get(s, mkt["base"] + mkt["klines"],
                     {"symbol": sym, "interval": "1d", "limit": 31}, mkt)
        d = d[:-1]  # drop today's unfinished candle
        if len(d) < a.min_age_days:
            continue
        hi = np.array([float(k[2]) for k in d])
        lo = np.array([float(k[3]) for k in d])
        qv = np.array([float(k[7]) for k in d])
        rng = hi / lo - 1
        rows.append({
            "symbol": sym,
            "vol24_usd": round(v),
            "med_daily_range": round(float(np.median(rng)), 4),
            "max_daily_range": round(float(rng.max()), 4),
            "med_daily_vol_usd": round(float(np.median(qv))),
            "days_of_data": len(d),
        })
        time.sleep(a.sleep)

    rows = [r for r in rows if r["med_daily_range"] >= a.min_daily_range]
    rows.sort(key=lambda r: r["med_daily_range"], reverse=True)
    rows = rows[: a.top]

    path = f"universe_{a.market}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved {len(rows)} symbols to {path}. Edit the file by hand if you want to add/remove coins.")


# ---------------------------------------------------------------- fetch

def cmd_fetch(a):
    mkt = MARKETS[a.market]
    with open(f"universe_{a.market}.json") as f:
        syms = [r["symbol"] for r in json.load(f)]
    if a.symbols:
        syms = a.symbols.split(",")
    outdir = os.path.join("data", a.market)
    os.makedirs(outdir, exist_ok=True)

    s = requests.Session()
    now_ms = int(time.time() * 1000) // 60000 * 60000
    end_ms = now_ms - 1  # only completed candles
    start_default = now_ms - a.days * 86400_000

    for n, sym in enumerate(syms, 1):
        path = os.path.join(outdir, f"{sym}_1m.csv.gz")
        old = None
        start = start_default
        if os.path.exists(path):
            old = pd.read_csv(path)
            start = max(start, int(old["t"].max()) + 60000)
        rows = []
        cur = start
        while cur < end_ms:
            data = http_get(s, mkt["base"] + mkt["klines"],
                            {"symbol": sym, "interval": "1m", "startTime": cur,
                             "endTime": end_ms, "limit": mkt["limit"]}, mkt)
            if not data:
                break
            rows.extend(data)
            cur = data[-1][0] + 60000
            if len(data) < mkt["limit"]:
                break
            time.sleep(a.sleep)
        if rows:
            df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 7, 8, 10]]
            df.columns = ["t", "o", "h", "l", "c", "qv", "n", "tbq"]
            df = df.astype({"t": "int64", "o": float, "h": float, "l": float, "c": float,
                            "qv": float, "n": "int64", "tbq": float})
            if old is not None:
                df = pd.concat([old, df])
            df = df.drop_duplicates("t").sort_values("t")
            df.to_csv(path, index=False, compression="gzip")
            total = len(df)
        else:
            total = 0 if old is None else len(old)
        print(f"[{n}/{len(syms)}] {sym}: +{len(rows)} candles, {total} total")


# ---------------------------------------------------------------- analysis helpers

def load_candles(path):
    df = pd.read_csv(path).drop_duplicates("t").sort_values("t")
    idx = np.arange(df["t"].iloc[0], df["t"].iloc[-1] + 60000, 60000)
    df = df.set_index("t").reindex(idx)
    df["c"] = df["c"].ffill()
    for col in ("o", "h", "l"):
        df[col] = df[col].fillna(df["c"])
    for col in ("qv", "n", "tbq"):
        df[col] = df[col].fillna(0)
    return df


def sim_long(o, h, l, c, e, tp, sl, hold):
    entry = o[e]
    tpp, slp = entry * (1 + tp), entry * (1 - sl)
    for k in range(e, e + hold):
        if o[k] <= slp:          # gapped through the stop
            return o[k] / entry - 1
        if l[k] <= slp:          # if stop and target are both hit in one candle, assume stop first
            return -sl
        if h[k] >= tpp:
            return tp
    return c[e + hold - 1] / entry - 1


def sim_short(o, h, l, c, e, tp, sl, hold):
    entry = o[e]
    tpp, slp = entry * (1 - tp), entry * (1 + sl)
    for k in range(e, e + hold):
        if o[k] >= slp:
            return 1 - o[k] / entry
        if h[k] >= slp:
            return -sl
        if l[k] <= tpp:
            return tp
    return 1 - c[e + hold - 1] / entry


def detect_events(df, a):
    qv = df["qv"]
    W = a.base_window
    med = qv.shift(1).rolling(W, min_periods=W // 2).median()
    mult = qv / med.clip(lower=1.0)
    ret1 = df["c"] / df["o"] - 1
    ret3 = df["c"] / df["c"].shift(3) - 1
    tbr = (df["tbq"] / qv).where(qv > 0)
    trig = ((mult >= a.vol_mult)
            & ((ret1 >= a.min_ret1) | (ret3 >= a.min_ret3))
            & (tbr >= a.min_tbr)
            & (qv >= a.min_minute_usd))
    idx = np.flatnonzero(trig.to_numpy())
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last >= a.cooldown:
            keep.append(i)
            last = i
    return keep, mult.to_numpy(), ret1.to_numpy(), ret3.to_numpy(), tbr.to_numpy(), med.to_numpy()


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "n=0"
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    pf = pos / neg if neg > 0 else float("inf")
    return (f"n={len(x):5d}  win={np.mean(x > 0)*100:5.1f}%  avg={x.mean()*100:+6.2f}%  "
            f"median={np.median(x)*100:+6.2f}%  sum={x.sum()*100:+8.1f}%  "
            f"PF={pf:4.2f}  worst={x.min()*100:+6.1f}%  best={x.max()*100:+6.1f}%")


# ---------------------------------------------------------------- analyze

def cmd_analyze(a):
    mkt = MARKETS[a.market]
    fee = mkt["fee"] if a.fee is None else a.fee
    cost = 2 * (fee + a.slip)
    datadir = os.path.join("data", a.market)
    files = sorted(f for f in os.listdir(datadir) if f.endswith("_1m.csv.gz"))
    if not files:
        sys.exit(f"No data in {datadir}. Run fetch first.")

    combos = [(tp, sl, hd) for tp in GRID_TP for sl in GRID_SL for hd in GRID_HOLD]
    need_after = max(max(HORIZONS), a.hold, max(GRID_HOLD), a.fade_window + a.hold, a.fade_window + max(GRID_HOLD)) + 2

    events, grid_long, grid_fade = [], [], []
    coin_days = 0.0
    for f in files:
        sym = f.replace("_1m.csv.gz", "")
        df = load_candles(os.path.join(datadir, f))
        coin_days += len(df) / 1440
        o, h, l, c = (df[k].to_numpy() for k in ("o", "h", "l", "c"))
        keep, mult, ret1, ret3, tbr, med = detect_events(df, a)
        t = df.index.to_numpy()
        for i in keep:
            if i < 60 or i + need_after >= len(df):
                continue
            e = i + 1
            entry = o[e]
            ev = {
                "symbol": sym,
                "time_utc": datetime.fromtimestamp(t[i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "t": int(t[i]),
                "vol_mult": round(mult[i], 1),
                "ret1": ret1[i], "ret3": ret3[i], "taker_buy_ratio": tbr[i],
                "minute_usd": df["qv"].iat[i], "baseline_minute_usd": med[i],
                "pre60": c[i] / c[i - 60] - 1,
                "entry_chase": entry / df["o"].iat[i] - 1,  # how far price already ran inside the trigger candle
            }
            for hz in HORIZONS:
                ev[f"fwd{hz}"] = c[e + hz - 1] / entry - 1
            win = slice(e, e + 15)
            ev["mfe15"] = h[win].max() / entry - 1
            ev["mae15"] = l[win].min() / entry - 1
            ev["mins_to_peak15"] = int(np.argmax(h[win])) + 1
            ev["long_net"] = sim_long(o, h, l, c, e, a.tp, a.sl, a.hold) - cost
            grid_long.append([sim_long(o, h, l, c, e, tp, sl, hd) - cost for tp, sl, hd in combos])

            # fade: after a big enough pump, short once price pulls back from the peak
            ev["fade_net"] = np.nan
            ev["fade_pump"] = np.nan
            ev["fade_delay"] = np.nan
            start_px = min(o[i], c[i - 3])
            peak = h[i]
            fg = [np.nan] * len(combos)
            for j in range(i, i + a.fade_window):
                peak = max(peak, h[j])
                pump = peak / start_px - 1
                if pump >= a.fade_min_pump and c[j] <= peak * (1 - a.fade_pullback):
                    ev["fade_pump"] = pump
                    ev["fade_delay"] = j - i
                    ev["fade_net"] = sim_short(o, h, l, c, j + 1, a.tp, a.sl, a.hold) - cost
                    fg = [sim_short(o, h, l, c, j + 1, tp, sl, hd) - cost for tp, sl, hd in combos]
                    break
            grid_fade.append(fg)
            events.append(ev)

    if not events:
        sys.exit("No events found. Loosen the thresholds (e.g. --vol-mult 5 --min-ret1 0.01).")

    ev = pd.DataFrame(events)
    # how many coins triggered within 2 minutes of each event (market-wide moves vs single-coin pumps)
    ts = np.sort(ev["t"].to_numpy())
    ev["concurrent"] = [int(np.searchsorted(ts, x + 120001) - np.searchsorted(ts, x - 120000)) for x in ev["t"]]
    cut = ev["t"].quantile(1 - a.oos_frac)
    ev["sample"] = np.where(ev["t"] <= cut, "in", "out")
    GL, GF = np.array(grid_long), np.array(grid_fade, dtype=float)

    os.makedirs("results", exist_ok=True)
    ev.drop(columns=["t"]).to_csv(f"results/events_{a.market}.csv", index=False)

    out = []
    p = out.append
    p(f"=== Pump event study ({a.market}) ===")
    p(f"Coins: {len(files)}   coin-days: {coin_days:.0f}   events: {len(ev)}   "
      f"events per coin per week: {len(ev) / coin_days * 7:.2f}")
    p(f"Trigger: vol >= {a.vol_mult}x {a.base_window}m median, 1m ret >= {a.min_ret1:.1%} or 3m ret >= {a.min_ret3:.1%}, "
      f"taker buy >= {a.min_tbr:.0%}, minute volume >= ${a.min_minute_usd:,.0f}, cooldown {a.cooldown}m")
    p(f"Entry: next candle open. Cost per round trip: {cost:.2%} (fee {fee:.2%} + slippage {a.slip:.2%}, per side)")
    p(f"Median chase inside trigger candle (already missed): {ev['entry_chase'].median():.2%}")
    p(f"In-sample until {datetime.fromtimestamp(cut/1000, tz=timezone.utc):%Y-%m-%d %H:%M} UTC, rest is out-of-sample")

    p("\n--- Raw forward returns from entry (gross, no costs) ---")
    for hz in HORIZONS:
        x = ev[f"fwd{hz}"]
        p(f"  +{hz:2d}m  mean {x.mean():+7.2%}  median {x.median():+7.2%}  up {np.mean(x > 0):5.1%}  "
          f"p10 {x.quantile(.1):+7.2%}  p90 {x.quantile(.9):+7.2%}")

    p("\n--- How often does the big move actually show up? (within 15m of entry) ---")
    for th in (0.03, 0.05, 0.10, 0.20, 0.50):
        p(f"  high reaches +{th:4.0%}: {np.mean(ev['mfe15'] >= th):6.1%}      "
          f"low reaches -{th:4.0%}: {np.mean(ev['mae15'] <= -th):6.1%}")
    p(f"  median minutes to the 15m peak: {ev['mins_to_peak15'].median():.0f}")

    def block(title, col):
        p(f"\n--- {title} ---")
        p("  all  " + stats(ev[col]))
        p("  in   " + stats(ev.loc[ev['sample'] == 'in', col]))
        p("  out  " + stats(ev.loc[ev['sample'] == 'out', col]))
        p("  solo (only coin moving)  " + stats(ev.loc[ev['concurrent'] == 1, col]))
        p("  crowd (2+ coins moving)  " + stats(ev.loc[ev['concurrent'] > 1, col]))

    block(f"LONG the spike: TP {a.tp:.0%} / SL {a.sl:.0%} / max hold {a.hold}m, net of costs", "long_net")
    block(f"FADE (short) after pump >= {a.fade_min_pump:.0%} and pullback {a.fade_pullback:.0%}: "
          f"TP {a.tp:.0%} / SL {a.sl:.0%} / hold {a.hold}m, net", "fade_net")
    if a.market == "spot":
        p("  (fade needs a perp contract; on spot this is only informational)")

    p("\n--- Concentration: top 8 coins by total LONG net ---")
    g = ev.groupby("symbol")["long_net"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
    p(g.head(8).to_string(float_format=lambda v: f"{v:+.3f}"))
    p("\n--- LONG net by month ---")
    ev["month"] = ev["time_utc"].str[:7]
    p(ev.groupby("month")["long_net"].agg(["count", "sum", "mean"]).to_string(float_format=lambda v: f"{v:+.3f}"))

    ins = (ev["sample"] == "in").to_numpy()
    rows = []
    for k, (tp, sl, hd) in enumerate(combos):
        for name, G in (("long", GL), ("fade", GF)):
            xi, xo = G[ins, k], G[~ins, k]
            xi, xo = xi[~np.isnan(xi)], xo[~np.isnan(xo)]
            rows.append({"side": name, "tp": tp, "sl": sl, "hold": hd,
                         "in_n": len(xi), "in_avg": xi.mean() if len(xi) else np.nan,
                         "in_win": (xi > 0).mean() if len(xi) else np.nan,
                         "out_n": len(xo), "out_avg": xo.mean() if len(xo) else np.nan,
                         "out_win": (xo > 0).mean() if len(xo) else np.nan})
    grid = pd.DataFrame(rows).sort_values(["side", "in_avg"], ascending=[True, False])
    grid.to_csv(f"results/grid_{a.market}.csv", index=False)
    for side in ("long", "fade"):
        p(f"\n--- Grid ({side}): best 8 by IN-sample avg, with OUT-of-sample next to it ---")
        sub = grid[grid["side"] == side].head(8)
        p(sub.drop(columns="side").to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    p("\nIf out-of-sample avg is much worse than in-sample, the 'best' setting is mostly luck.")

    text = "\n".join(out)
    print(text)
    with open(f"results/summary_{a.market}.txt", "w") as f:
        f.write(text + "\n")
    print(f"\nWrote results/summary_{a.market}.txt, results/events_{a.market}.csv, results/grid_{a.market}.csv")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("universe", help="pick coins")
    u.add_argument("--market", choices=MARKETS, default="spot")
    u.add_argument("--min-vol24", type=float, default=2e6, help="min 24h volume in USDT (enough liquidity to trade)")
    u.add_argument("--max-vol24", type=float, default=150e6, help="max 24h volume (keep out big, hard-to-pump coins)")
    u.add_argument("--min-daily-range", type=float, default=0.08, help="min median daily high/low range")
    u.add_argument("--min-age-days", type=int, default=20, help="skip coins with fewer days of history")
    u.add_argument("--top", type=int, default=60, help="keep this many most volatile coins")
    u.add_argument("--sleep", type=float, default=0.1)

    fe = sub.add_parser("fetch", help="download 1m candles")
    fe.add_argument("--market", choices=MARKETS, default="spot")
    fe.add_argument("--days", type=int, default=90)
    fe.add_argument("--symbols", help="comma list to override the universe file")
    fe.add_argument("--sleep", type=float, default=0.15)

    an = sub.add_parser("analyze", help="event study")
    an.add_argument("--market", choices=MARKETS, default="spot")
    an.add_argument("--base-window", type=int, default=60, help="minutes of baseline volume")
    an.add_argument("--vol-mult", type=float, default=8, help="minute volume vs baseline median")
    an.add_argument("--min-ret1", type=float, default=0.015, help="trigger candle return")
    an.add_argument("--min-ret3", type=float, default=0.03, help="or 3-minute return")
    an.add_argument("--min-tbr", type=float, default=0.6, help="taker buy share of volume")
    an.add_argument("--min-minute-usd", type=float, default=30000, help="trigger minute must trade at least this much")
    an.add_argument("--cooldown", type=int, default=60, help="min minutes between events on one coin")
    an.add_argument("--tp", type=float, default=0.05)
    an.add_argument("--sl", type=float, default=0.03)
    an.add_argument("--hold", type=int, default=15)
    an.add_argument("--fee", type=float, default=None, help="per side; default spot 0.1%%, futures 0.05%%")
    an.add_argument("--slip", type=float, default=0.002, help="slippage per side")
    an.add_argument("--fade-min-pump", type=float, default=0.08)
    an.add_argument("--fade-pullback", type=float, default=0.03)
    an.add_argument("--fade-window", type=int, default=30)
    an.add_argument("--oos-frac", type=float, default=0.3, help="last share of events kept as out-of-sample")

    a = ap.parse_args()
    {"universe": cmd_universe, "fetch": cmd_fetch, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
