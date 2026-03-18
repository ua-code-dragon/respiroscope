#!/usr/bin/env python3

from __future__ import annotations

import os, sys
import csv
import math
import json
import time as ts
from datetime import datetime, date, timedelta, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib
matplotlib.use('Agg') # for MacOs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from anova import one_way_anova, OneWayAnovaResult

max_workers = 16
min_lines = 48
wd = Path(os.path.realpath(__file__)).parent
root = Path(os.path.join(wd,'work','stat'))
imgs = Path(os.path.join(wd,'work','imgs'))

results = []

def rprint(s):
    print(s,end="")
    sys.stdout.flush()

def iter_files(root: Path) -> Iterable[Path]:
    for p in root.glob('**/*.csv'):
        if p.is_file():
            yield p

def draw_ts_plot(sid, fio2, spo2):
    out_path = Path(os.path.join(imgs,f'{sid}_ts.png'))
    fig, ax_left = plt.subplots()
    line1, = ax_left.plot(range(len(fio2)), fio2, label='FiO2', color="red", linewidth=2)
    ax_left.set_xlabel("Time, h")
    ax_left.set_ylabel("FiO2")
    ax_right = ax_left.twinx()
    line2, = ax_right.plot(range(len(spo2)), spo2, label="SpO2", color="black", linewidth=2)
    ax_right.set_ylabel("SpO2")
    ax_left.legend(handles=[line1, line2], loc="upper right", facecolor="white", framealpha=1.0,)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def draw_sc_plot(sid, fio2, spo2, ts):
    out_path = Path(os.path.join(imgs,f'{sid}_sc.png'))
    vmin = min(ts)
    vmax = max(ts)
    if vmin == vmax:
        vmax = vmin + 1
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)        
    fig, ax = plt.subplots()
    sc = ax.scatter(
    fio2,
    spo2,
    c=ts,
    cmap='viridis',
    norm=norm,
    s=30, # marker size
    alpha=0.9,
    edgecolors="none",
    label=None,
    )
    ax.set_xlabel("FiO2")
    ax.set_ylabel("SpO2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def parse_file(path: Path):
    rprint("-")
    df = pd.read_csv(path)
    fio2 = df['fio2'].tolist()
    spo2 = df['spo2'].tolist()
    sid = path.stem
    size = len(fio2)
    try:
        res: OneWayAnovaResult = one_way_anova( fio2, spo2 )
    except Exception as e:
        #print(f"ANOVA error: {e}")
        rprint("!")
        return {"ok": False}
    
    if size > 143 and res.k > 4:
        draw_ts_plot(sid, fio2, spo2)
        draw_sc_plot(sid, fio2, spo2, range(len(fio2)))
    
    rprint("|")
    return { "ok": True, "stat": {
        "id": sid,
        "size": size,
        "n": res.N,
        "k": res.k,
        "ssb": res.ss_between,
        "ssw": res.ss_within,
        "f": res.F,
        "p": res.p_value,
        "eta2": res.eta2,
        "omega2": res.omega2,
    }}


if __name__ == "__main__":                    

    _st = ts.time()

    if not os.path.isdir(imgs):
        os.makedirs(imgs)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = (pool.submit(parse_file, p) for p in iter_files(root))
        for fut in as_completed(futures):
            results.append(fut.result())

    print("\n")

    stat = [r['stat'] for r in results if r['ok']]
    
    with Path(os.path.join(root.parent,'stat_anova.csv')).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f,fieldnames=["id", "size", "n", "k", "ssb", "ssw", "f", "p", "eta2", "omega2"])
        writer.writeheader()
        writer.writerows(stat)
    
    
    print("RES: ",len(stat))

    _et = ts.time()

    print(f"ETA: {timedelta(seconds=(_et-_st))}")




