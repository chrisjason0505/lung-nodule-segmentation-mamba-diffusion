"""Build the nodule-crop cache from raw LIDC-IDRI DICOM + pylidc annotations.

For every nodule (cluster of radiologist annotations, via pylidc) annotated by
at least ``--min_readers`` radiologists we store a ``crop x crop x crop`` cube,
resampled to ``--spacing`` mm isotropic and centred on the 50% consensus mask:

    image    int16 (D,H,W)   Hounsfield units
    mask     uint8 (D,H,W)   50% consensus of the readers (pylidc convention)
    readers  uint8 (R,D,H,W) each radiologist's own mask (R = 1..4)

Optionally, a few random lung cubes per scan with no label (``--unlabeled_per_scan``)
are stored too; they are only used for self-supervised denoising pretraining.

Usage
-----
    python -m lungseg.data.prepare_lidc --dicom_root /path/to/LIDC-IDRI --out data/lidc_crops

The DICOM root can be any folder that contains the TCIA series (the Kaggle
mirror, an NBIA download, ...). Splits are made per patient, so no patient is
in more than one of train/val/test.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from scipy import ndimage

from lungseg.compat import import_pylidc
from lungseg.data.dicom_io import LazySeries, build_series_index

META_FIELDS = ["file", "patient_id", "scan_id", "nodule_idx", "n_readers", "diameter_mm",
               "volume_mm3", "malignancy", "subtlety", "texture", "calcification", "spiculation",
               "slice_thickness", "pixel_spacing", "center_row", "center_col", "center_z",
               "mask_voxels"]


def _grid(center_mm, size, spacing):
    off = (np.arange(size) - (size - 1) / 2.0) * spacing
    return center_mm + off


def resample_cube(series: LazySeries, slice_z: np.ndarray, pixel_spacing: float,
                  center_rcz: tuple, size: int, spacing: float, reader_masks=None,
                  hu_fill: float = -1024.0):
    """Trilinear resample of an isotropic cube around ``center_rcz`` = (row_mm, col_mm, z_mm).

    ``slice_z``: z position (mm) of pylidc slice index k, ascending.
    ``reader_masks``: list of (mask_bool, (si, sj, sk)) in full-scan pylidc index space.
    Returns image (D,H,W) float32, and reader masks (R,D,H,W) float32 in [0,1].
    """
    rows_mm = _grid(center_rcz[0], size, spacing)
    cols_mm = _grid(center_rcz[1], size, spacing)
    z_mm = _grid(center_rcz[2], size, spacing)

    n_k = len(slice_z)
    fk = np.interp(z_mm, slice_z, np.arange(n_k), left=-10.0, right=n_k + 9.0)
    fr = rows_mm / pixel_spacing
    fc = cols_mm / pixel_spacing

    # native sub-block (with 1 voxel margin for interpolation)
    k0, k1 = max(int(np.floor(fk.min())) - 1, 0), min(int(np.ceil(fk.max())) + 1, n_k - 1)
    r0, r1 = max(int(np.floor(fr.min())) - 1, 0), min(int(np.ceil(fr.max())) + 1, series.rows - 1)
    c0, c1 = max(int(np.floor(fc.min())) - 1, 0), min(int(np.ceil(fc.max())) + 1, series.cols - 1)
    if k1 < k0 or r1 < r0 or c1 < c0:
        raise ValueError("cube lies completely outside the scan")

    block = np.empty((k1 - k0 + 1, r1 - r0 + 1, c1 - c0 + 1), np.float32)
    for kk in range(k0, k1 + 1):
        fidx = series.index_of_z(slice_z[kk])
        block[kk - k0] = series.slice_hu(fidx)[r0:r1 + 1, c0:c1 + 1]

    K, R, C = np.meshgrid(fk - k0, fr - r0, fc - c0, indexing="ij")
    coords = np.stack([K, R, C])
    image = ndimage.map_coordinates(block, coords, order=1, mode="constant", cval=hu_fill)

    readers = []
    for mask, (si, sj, sk) in (reader_masks or []):
        m = np.zeros_like(block, dtype=np.float32)
        # mask is (i=row, j=col, k=slice); block is (k,row,col)
        mt = np.transpose(mask.astype(np.float32), (2, 0, 1))
        ks, ke = sk.start - k0, sk.stop - k0
        rs, re = si.start - r0, si.stop - r0
        cs, ce = sj.start - c0, sj.stop - c0
        # clip to block
        dk0, dr0, dc0 = max(0, -ks), max(0, -rs), max(0, -cs)
        ks, rs, cs = max(ks, 0), max(rs, 0), max(cs, 0)
        ke, re, ce = min(ke, m.shape[0]), min(re, m.shape[1]), min(ce, m.shape[2])
        if ke > ks and re > rs and ce > cs:
            m[ks:ke, rs:re, cs:ce] = mt[dk0:dk0 + ke - ks, dr0:dr0 + re - rs, dc0:dc0 + ce - cs]
        readers.append(ndimage.map_coordinates(m, coords, order=1, mode="constant", cval=0.0))
    readers = np.stack(readers) if readers else np.zeros((0,) + image.shape, np.float32)
    return image.astype(np.float32), readers


def process_scan(scan_id: int, series_dir: str, out_dir: str, size: int, spacing: float,
                 min_readers: int, unlabeled_per_scan: int, seed: int):
    pl = import_pylidc()
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    series = LazySeries.open(series_dir)
    slice_z = np.asarray(scan.slice_zvals, np.float64)
    order = np.argsort(slice_z)
    if not np.all(order == np.arange(len(order))):
        raise ValueError("pylidc slice_zvals are not ascending")
    ps = float(scan.pixel_spacing)
    records, n_skipped = [], 0

    clusters = scan.cluster_annotations(verbose=False)
    for ni, anns in enumerate(clusters):
        if len(anns) < min_readers:
            n_skipped += 1
            continue
        masks = [(a.boolean_mask(), a.bbox()) for a in anns]
        # centroid of the 50% consensus in native index space (i,j,k)
        acc = {}
        for m, (si, sj, sk) in masks:
            idx = np.argwhere(m) + np.array([si.start, sj.start, sk.start])
            for t in map(tuple, idx):
                acc[t] = acc.get(t, 0) + 1
        pts = np.array([t for t, c in acc.items() if c / len(anns) >= 0.5] or list(acc.keys()), float)
        ci, cj, ck = pts.mean(0)
        cz = float(np.interp(ck, np.arange(len(slice_z)), slice_z))
        center = (ci * ps, cj * ps, cz)

        image, readers = resample_cube(series, slice_z, ps, center, size, spacing, masks)
        readers_bin = (readers >= 0.5).astype(np.uint8)
        consensus = (readers.mean(0) >= 0.5).astype(np.uint8)
        if consensus.sum() == 0:  # extremely small nodule vanished after resampling
            consensus = (readers.max(0) >= 0.5).astype(np.uint8)
        fname = f"{scan.patient_id}_s{scan.id:04d}_n{ni:02d}.npz"
        np.savez_compressed(os.path.join(out_dir, "labeled", fname),
                            image=np.clip(np.round(image), -1024, 3071).astype(np.int16),
                            mask=consensus, readers=readers_bin,
                            spacing=np.float32(spacing))

        def mean_attr(name):
            return float(np.mean([getattr(a, name) for a in anns]))
        records.append(dict(
            file=f"labeled/{fname}", patient_id=scan.patient_id, scan_id=scan.id, nodule_idx=ni,
            n_readers=len(anns), diameter_mm=mean_attr("diameter"), volume_mm3=mean_attr("volume"),
            malignancy=mean_attr("malignancy"), subtlety=mean_attr("subtlety"),
            texture=mean_attr("texture"), calcification=mean_attr("calcification"),
            spiculation=mean_attr("spiculation"), slice_thickness=float(scan.slice_thickness),
            pixel_spacing=ps, center_row=center[0], center_col=center[1], center_z=center[2],
            mask_voxels=int(consensus.sum())))

    # unlabeled lung cubes (self-supervised pretraining only)
    unl = []
    rng = np.random.default_rng(seed + int(scan.id))
    for u in range(unlabeled_per_scan):
        for _try in range(10):
            k = int(rng.integers(int(0.2 * len(slice_z)), max(int(0.8 * len(slice_z)), 1)))
            sl = series.slice_hu(series.index_of_z(slice_z[k]))
            lung = np.argwhere((sl > -950) & (sl < -400))
            if len(lung) < 500:
                continue
            r, c = lung[rng.integers(len(lung))]
            image, _ = resample_cube(series, slice_z, ps, (r * ps, c * ps, slice_z[k]), size, spacing)
            fname = f"{scan.patient_id}_s{scan.id:04d}_u{u:02d}.npz"
            np.savez_compressed(os.path.join(out_dir, "unlabeled", fname),
                                image=np.clip(np.round(image), -1024, 3071).astype(np.int16),
                                spacing=np.float32(spacing))
            unl.append(dict(file=f"unlabeled/{fname}", patient_id=scan.patient_id, scan_id=scan.id))
            break
    series.clear()
    return records, unl, n_skipped


def make_splits(patient_ids, seed=0, val_frac=0.1, test_frac=0.2):
    pids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_test = int(round(test_frac * len(pids)))
    n_val = int(round(val_frac * len(pids)))
    return {"test": sorted(pids[:n_test]), "val": sorted(pids[n_test:n_test + n_val]),
            "train": sorted(pids[n_test + n_val:]), "seed": seed}


def _worker(args):
    try:
        return args[0], process_scan(*args), None
    except Exception:
        return args[0], None, traceback.format_exc()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom_root", required=True)
    ap.add_argument("--out", default="data/lidc_crops")
    ap.add_argument("--size", type=int, default=96, help="stored cube edge (voxels)")
    ap.add_argument("--spacing", type=float, default=0.8, help="isotropic spacing (mm)")
    ap.add_argument("--min_readers", type=int, default=2)
    ap.add_argument("--unlabeled_per_scan", type=int, default=2)
    ap.add_argument("--max_scans", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(os.path.join(args.out, "labeled"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "unlabeled"), exist_ok=True)
    index = build_series_index(args.dicom_root, os.path.join(args.out, "series_index.json"))

    pl = import_pylidc()
    scans = [(s.id, s.patient_id, s.series_instance_uid) for s in pl.query(pl.Scan).all()]
    jobs = [(sid, index[uid]) for sid, _pid, uid in scans if uid in index]
    print(f"[prepare] {len(jobs)}/{len(scans)} pylidc scans have DICOM data under {args.dicom_root}")
    if args.max_scans:
        jobs = jobs[:args.max_scans]
    if not jobs:
        raise SystemExit("No scans found. Point --dicom_root at the folder containing LIDC-IDRI-XXXX dirs.")

    tasks = [(sid, d, args.out, args.size, args.spacing, args.min_readers,
              args.unlabeled_per_scan, args.seed) for sid, d in jobs]
    records, unlabeled, failures, skipped = [], [], [], 0
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                sid, res, err = f.result()
                if err:
                    failures.append((sid, err))
                else:
                    records += res[0]; unlabeled += res[1]; skipped += res[2]
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} scans, {len(records)} nodules, "
                          f"{len(failures)} failed, {time.time() - t0:.0f}s", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            sid, res, err = _worker(t)
            if err:
                failures.append((sid, err))
            else:
                records += res[0]; unlabeled += res[1]; skipped += res[2]
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} scans, {len(records)} nodules, {time.time() - t0:.0f}s", flush=True)

    records.sort(key=lambda r: r["file"])
    with open(os.path.join(args.out, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(args.out, "unlabeled.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "patient_id", "scan_id"])
        w.writeheader()
        w.writerows(sorted(unlabeled, key=lambda r: r["file"]))
    splits = make_splits([r["patient_id"] for r in records], seed=args.seed)
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    if failures:
        with open(os.path.join(args.out, "failures.txt"), "w") as f:
            for sid, err in failures:
                f.write(f"scan {sid}\n{err}\n")
    print(f"[prepare] done: {len(records)} nodules from {len({r['patient_id'] for r in records})} patients "
          f"({skipped} clusters with < {args.min_readers} readers skipped), {len(unlabeled)} unlabeled cubes, "
          f"{len(failures)} scans failed. Splits (patients): "
          f"train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}")


if __name__ == "__main__":
    main()
