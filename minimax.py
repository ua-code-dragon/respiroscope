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

max_workers = 16
min_lines = 48
wd = Path(os.path.realpath(__file__)).parent
root = Path(os.path.join(wd,'subsets','stat'))
dest = Path(os.path.join(wd,'work','stat'))

results = []

def rprint(s):
    print(s,end="")
    sys.stdout.flush()

def iter_files(root: Path) -> Iterable[Path]:
    for p in root.glob('**/*.csv'):
        if p.is_file():
            yield p

def parse_file(path: Path):
    badfile = False
    res = []
    id = None
    fio2_max = None
    fio2_min = None
    spo2_max = None
    spo2_min = None
    fio2_avg = None
    fio2_var = None
    spo2_avg = None
    spo2_var = None
    fio2_n = 0
    spo2_n = 0
    fio2_d = 0
    spo2_d = 0
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n += 1
            if int(r.get('invasive','0')) == 1:
                badfile = True
                break
            id = r.get('subject_id')
            _fio2 = r.get('set_fio2')
            _fio2 = float(_fio2) if _fio2 else None
            _spo2 = r.get('spo2')            
            _spo2 = float(_spo2) if _spo2 else None
            if _spo2 is not None:
                spo2_n += 1
                spo2_d = _spo2 - spo2_avg if spo2_avg else _spo2
                spo2_avg = (spo2_avg or 0) + spo2_d/spo2_n
                spo2_var = (spo2_var or 0) + spo2_d * (_spo2 - spo2_avg)
                spo2_min = _spo2 if spo2_min is None or _spo2 < spo2_min else spo2_min
                spo2_max = _spo2 if spo2_max is None or _spo2 > spo2_max else spo2_max
            if _fio2 is not None:    
                fio2_n += 1
                fio2_d = _fio2 - fio2_avg if fio2_avg else _fio2
                fio2_avg = (fio2_avg or 0) + fio2_d/fio2_n
                fio2_var = (fio2_var or 0) + fio2_d * (_fio2 - fio2_avg)
                fio2_min = _fio2 if fio2_min is None or _fio2 < fio2_min else fio2_min
                fio2_max = _fio2 if fio2_max is None or _fio2 > fio2_max else fio2_max
    if badfile:
        return {'ok':False}
    return {
    'ok': True, 'stat': {'id':id, 'size':n,
    'spo2_min':spo2_min,'spo2_max':spo2_max,'spo2_avg':spo2_avg,'spo2_var':spo2_var,
    'fio2_min':fio2_min,'fio2_max':fio2_max,'fio2_avg':fio2_avg,'fio2_var':fio2_var,
    }}     

if __name__ == "__main__":                    

    _st = ts.time()

    if not os.path.isdir(dest):
        os.makedirs(dest)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = (pool.submit(parse_file, p) for p in iter_files(root))
        for fut in as_completed(futures):
            results.append(fut.result())
    
    stat = [r['stat'] for r in results if r['ok']]

    with Path(os.path.join(dest,'minmax.csv')).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f,fieldnames=['id','size','spo2_min','spo2_max','spo2_avg','spo2_var','fio2_min','fio2_max','fio2_avg','fio2_var'])
        writer.writeheader()
        writer.writerows(stat)
    
    print("RES: ",len([r for r in results if r['ok']]))

    _et = ts.time()

    print(f"ETA: {timedelta(seconds=(_et-_st))}")




