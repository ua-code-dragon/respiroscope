#!/usr/bin/env python3

from __future__ import annotations

import os, sys
import csv
import math
import time as ts
from datetime import datetime, date, timedelta, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

max_workers = 16
min_lines = 48
wd = Path(os.path.realpath(__file__)).parent
root = Path(os.path.join(wd,'data110'))
dest = Path(os.path.join(wd,'subsets','stat'))

results = []


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.glob('**/*.csv'):
        if p.is_file():
            yield p

def parse_file(path: Path):
    def _data_lines(r):
        _reqf = ['los','set_fio2','spo2']
        edata = {'id': r.get('subject_id'),'age': r.get('age'),'len':0}
        if any(k not in r for k in _reqf):
            return False, 'Missing column', edata
        n = math.ceil(float(r['los']) * 24.0)    
        edata['len'] = n
        if n < min_lines:
            return False, f'Short file ({n})', edata
        return n, '', edata    
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        r = next(reader)
        n,des,edata = _data_lines(r)
        if not n:
            return {'ok':False,'err':des,'data':edata}
        dpath = Path(os.path.join(dest,path.name))
        os.system(f"head -n {n+1} {path} > {dpath}")
        return {'ok':True, 'data': edata}
        

if __name__ == "__main__":                    

    _st = ts.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = (pool.submit(parse_file, p) for p in iter_files(root))
        for fut in as_completed(futures):
            results.append(fut.result())

    ages = [r['data']['age'] for r in results if r['data']['age'] is not None]
    lens = [r['data']['len'] for r in results if r['data']['len'] is not None and r['data']['len'] > 0]

    with Path(os.path.join(dest.parent,'stat.age.dat')).open('w') as f:
        f.write("\n".join(list(map(str,ages))))
    with Path(os.path.join(dest.parent,'stat.len.dat')).open('w') as f:
        f.write("\n".join(list(map(str,lens))))


    print("RES: ",len([r for r in results if r['ok']]))

    _et = ts.time()

    print(f"ETA: {timedelta(seconds=(_et-_st))}")




