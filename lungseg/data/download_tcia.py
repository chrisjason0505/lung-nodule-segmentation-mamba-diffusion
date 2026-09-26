"""Download LIDC-IDRI CT series straight from TCIA (public, no login).

Use this only if you do not have the Kaggle mirror or an NBIA download. It
fetches only the CT series that pylidc has annotations for, one zip per series.

    python -m lungseg.data.download_tcia --out /content/LIDC-IDRI --n_patients 300

Full LIDC-IDRI is about 125 GB (1,018 CT series). ``--n_patients`` downloads a
random, reproducible subset of patients. The patient-level splits are made
later from whatever was downloaded.
"""
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from lungseg.compat import import_pylidc

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"


def fetch(uid, pid, out, retries=4):
    dest = os.path.join(out, pid, uid)
    if os.path.isdir(dest) and any(f.endswith(".dcm") for f in os.listdir(dest)):
        return uid, "exists"
    for a in range(retries):
        try:
            r = requests.get(API, params={"SeriesInstanceUID": uid}, timeout=600)
            r.raise_for_status()
            os.makedirs(dest, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(dest)
            return uid, "ok"
        except Exception as e:  # pragma: no cover - network
            err = str(e)
            time.sleep(5 * (a + 1))
    return uid, "failed: " + err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_patients", type=int, default=0, help="0 = all annotated patients")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pl = import_pylidc()
    scans = [(s.patient_id, s.series_instance_uid) for s in pl.query(pl.Scan).all() if len(s.annotations)]
    pids = sorted({p for p, _ in scans})
    if args.n_patients:
        pids = sorted(np.random.default_rng(args.seed).choice(pids, args.n_patients, replace=False))
    todo = [(u, p) for p, u in scans if p in set(pids)]
    print(f"downloading {len(todo)} series from {len(pids)} patients -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (uid, status) in enumerate(ex.map(lambda t: fetch(t[0], t[1], args.out), todo), 1):
            if status != "exists":
                print(f"  [{i}/{len(todo)}] {uid[-12:]} {status}", flush=True)


if __name__ == "__main__":
    main()
