#!/usr/bin/env python3
"""
plot_events.py - draw candles + volume for one coin, with the trigger, entry and exit
of every event from pump_study.py, so you can see what the strategy actually did.

Put this file next to pump_study.py and run it from the same folder.
Needs: pip install plotly

Examples:
  python plot_events.py ERAUSDT --list                 # list this coin's events with index numbers
  python plot_events.py ERAUSDT --event 3              # zoom on event #3 (90 min each side)
  python plot_events.py ERAUSDT --start 2026-10-15 --end 2026-10-18
  python plot_events.py ERAUSDT                        # last 3 days of data
  python plot_events.py ERAUSDT --event 3 --fade       # also draw the short (fade) trade

Times are shown in Pakistan time by default (--tz to change). Use the SAME --tp/--sl/--hold/
--fee/--slip values you used for analyze, otherwise the exits drawn here won't match the CSV.
"""
import argparse
import os
import sys
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pump_study import MARKETS, load_candles


def sim_long_detail(o, h, l, c, e, tp, sl, hold):
    entry = o[e]
    tpp, slp = entry * (1 + tp), entry * (1 - sl)
    for k in range(e, e + hold):
        if o[k] <= slp:
            return k, o[k], "stop (gap)"
        if l[k] <= slp:
            return k, slp, "stop"
        if h[k] >= tpp:
            return k, tpp, "target"
    k = e + hold - 1
    return k, c[k], "time"


def sim_short_detail(o, h, l, c, e, tp, sl, hold):
    entry = o[e]
    tpp, slp = entry * (1 - tp), entry * (1 + sl)
    for k in range(e, e + hold):
        if o[k] >= slp:
            return k, o[k], "stop (gap)"
        if h[k] >= slp:
            return k, slp, "stop"
        if l[k] <= tpp:
            return k, tpp, "target"
    k = e + hold - 1
    return k, c[k], "time"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol")
    ap.add_argument("--market", choices=MARKETS, default="spot")
    ap.add_argument("--list", action="store_true", help="print this coin's events and exit")
    ap.add_argument("--event", type=int, help="zoom on this event index (see --list)")
    ap.add_argument("--window", type=int, default=90, help="minutes each side when zooming on an event")
    ap.add_argument("--start", help="YYYY-MM-DD (in --tz)")
    ap.add_argument("--end", help="YYYY-MM-DD (in --tz), inclusive")
    ap.add_argument("--tz", default="Asia/Karachi")
    ap.add_argument("--fade", action="store_true", help="also draw the fade (short) trades")
    ap.add_argument("--tp", type=float, default=0.05)
    ap.add_argument("--sl", type=float, default=0.03)
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--slip", type=float, default=0.002)
    ap.add_argument("--events", help="events CSV (default results/events_<market>.csv)")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    a = ap.parse_args()

    sym = a.symbol.upper()
    fee = MARKETS[a.market]["fee"] if a.fee is None else a.fee
    cost = 2 * (fee + a.slip)

    path = os.path.join("data", a.market, f"{sym}_1m.csv.gz")
    if not os.path.exists(path):
        sys.exit(f"No candle file {path}")
    df = load_candles(path)
    o, h, l, c = (df[k].to_numpy() for k in ("o", "h", "l", "c"))
    t_ms = df.index.to_numpy()
    local = pd.to_datetime(t_ms, unit="ms", utc=True).tz_convert(a.tz).tz_localize(None)

    ev_path = a.events or os.path.join("results", f"events_{a.market}.csv")
    ev = pd.read_csv(ev_path)
    ev = ev[ev["symbol"] == sym].reset_index(drop=True)

    # locate each event's trigger candle and rebuild its trades
    trades = []
    t_index = {int(v): n for n, v in enumerate(t_ms)}
    mismatch = 0
    for n, r in ev.iterrows():
        tm = int(pd.Timestamp(r["time_utc"], tz="UTC").value // 1_000_000)
        i = t_index.get(tm)
        if i is None:
            continue
        e = i + 1
        k, px, why = sim_long_detail(o, h, l, c, e, a.tp, a.sl, a.hold)
        net = px / o[e] - 1 - cost
        if abs(net - r["long_net"]) > 1e-6:
            mismatch += 1
        tr = {"n": n, "i": i, "e": e, "x": k, "entry": o[e], "exit": px, "why": why, "net": net, "row": r}
        if pd.notna(r.get("fade_delay")):
            fe_ = i + int(r["fade_delay"]) + 1
            fk, fpx, fwhy = sim_short_detail(o, h, l, c, fe_, a.tp, a.sl, a.hold)
            tr.update(f_e=fe_, f_x=fk, f_entry=o[fe_], f_exit=fpx, f_why=fwhy, f_net=1 - fpx / o[fe_] - cost)
        trades.append(tr)
    if mismatch:
        print(f"Note: {mismatch} trades differ from the CSV's long_net. "
              "Use the same --tp/--sl/--hold/--fee/--slip as your analyze run.")

    if a.list:
        if not trades:
            print("No events for this coin.")
        for tr in trades:
            r = tr["row"]
            f = f"  fade {tr['f_net']:+.2%}" if "f_net" in tr else ""
            print(f"#{tr['n']:3d}  {local[tr['i']]:%Y-%m-%d %H:%M}  vol x{r['vol_mult']:<7.0f} "
                  f"ret1 {r['ret1']:+.1%}  long {tr['net']:+.2%} ({tr['why']})  mfe15 {r['mfe15']:+.1%}{f}")
        return

    # choose the window to draw
    if a.event is not None:
        tr = next((t for t in trades if t["n"] == a.event), None)
        if tr is None:
            sys.exit("No such event index. Run with --list.")
        lo, hi = max(0, tr["i"] - a.window), min(len(df), tr["i"] + a.window)
        label = f"event{a.event}"
    else:
        if a.start or a.end:
            s = pd.Timestamp(a.start) if a.start else local[0]
            en = (pd.Timestamp(a.end) + pd.Timedelta(days=1)) if a.end else local[-1] + pd.Timedelta(minutes=1)
        else:
            en = local[-1] + pd.Timedelta(minutes=1)
            s = en - pd.Timedelta(days=3)
        lo, hi = int(np.searchsorted(local, s)), int(np.searchsorted(local, en))
        label = f"{s:%Y%m%d}-{en:%Y%m%d}"
    if hi - lo > 20000:
        print(f"Warning: {hi - lo} candles; the chart may be slow. Use --start/--end or --event.")
    if hi <= lo:
        sys.exit("Nothing in that range.")

    x = local[lo:hi]
    d = df.iloc[lo:hi]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(x=x, open=d["o"], high=d["h"], low=d["l"], close=d["c"], name=sym,
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    up = (d["c"] >= d["o"]).to_numpy()
    fig.add_trace(go.Bar(x=x, y=d["qv"], name="volume $", marker_color=np.where(up, "#26a69a", "#ef5350"),
                         opacity=0.55), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["tbq"], name="taker buy $", mode="lines",
                             line=dict(color="#f0b90b", width=1)), row=2, col=1)

    shown = [t for t in trades if lo <= t["i"] < hi]
    for tr in shown:
        r, col = tr["row"], ("#2e7d32" if tr["net"] > 0 else "#c62828")
        # trigger candle shading
        fig.add_vrect(x0=local[tr["i"]] - pd.Timedelta(seconds=30), x1=local[tr["i"]] + pd.Timedelta(seconds=30),
                      fillcolor="#f0b90b", opacity=0.25, line_width=0)
        # target / stop levels for the life of the trade
        xs = [local[tr["e"]], local[tr["x"]]]
        for lvl, dash in ((tr["entry"] * (1 + a.tp), "dot"), (tr["entry"] * (1 - a.sl), "dot")):
            fig.add_trace(go.Scatter(x=xs, y=[lvl, lvl], mode="lines", line=dict(color="#888", dash=dash, width=1),
                                     showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[tr["entry"], tr["exit"]], mode="lines",
                                 line=dict(color=col, width=2), showlegend=False, hoverinfo="skip"), row=1, col=1)
        info = (f"event #{tr['n']}<br>vol x{r['vol_mult']:.0f}, ret1 {r['ret1']:+.1%}, taker buy {r['taker_buy_ratio']:.0%}"
                f"<br>chase {r['entry_chase']:+.1%}, pre60 {r['pre60']:+.1%}")
        fig.add_trace(go.Scatter(x=[local[tr["e"]]], y=[tr["entry"]], mode="markers",
                                 marker=dict(symbol="triangle-up", size=13, color="#1565c0", line=dict(width=1, color="white")),
                                 name="long entry", legendgroup="le", showlegend=tr is shown[0],
                                 hovertext=info + f"<br>entry {tr['entry']:.6g}", hoverinfo="text"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[local[tr["x"]]], y=[tr["exit"]], mode="markers",
                                 marker=dict(symbol="x", size=12, color=col),
                                 name="long exit", legendgroup="lx", showlegend=tr is shown[0],
                                 hovertext=f"exit {tr['exit']:.6g} ({tr['why']})<br>net {tr['net']:+.2%}",
                                 hoverinfo="text"), row=1, col=1)
        if a.fade and "f_net" in tr:
            fcol = "#2e7d32" if tr["f_net"] > 0 else "#c62828"
            fig.add_trace(go.Scatter(x=[local[tr["f_e"]], local[tr["f_x"]]], y=[tr["f_entry"], tr["f_exit"]],
                                     mode="lines+markers", line=dict(color=fcol, width=2, dash="dash"),
                                     marker=dict(symbol=["triangle-down", "x"], size=12, color=["#e65100", fcol]),
                                     name="fade (short)", legendgroup="f", showlegend=tr is shown[0],
                                     hovertext=[f"short entry {tr['f_entry']:.6g}",
                                                f"cover {tr['f_exit']:.6g} ({tr['f_why']})<br>net {tr['f_net']:+.2%}"],
                                     hoverinfo="text"), row=1, col=1)

    wins = sum(t["net"] > 0 for t in shown)
    title = (f"{sym} {a.market} | {len(shown)} events in view, {wins} long wins | "
             f"TP {a.tp:.0%} SL {a.sl:.0%} hold {a.hold}m, cost {cost:.2%} | times {a.tz}")
    fig.update_layout(title=title, template="plotly_dark", height=850, xaxis_rangeslider_visible=False,
                      hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="USDT / min", row=2, col=1)

    os.makedirs("charts", exist_ok=True)
    out = os.path.join("charts", f"{sym}_{a.market}_{label}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}  ({len(shown)} events, {hi - lo} candles)")
    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
