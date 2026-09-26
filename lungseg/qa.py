"""Visual + numeric QA of the preprocessing against pylidc's own reference path.

1. Re-creates the official pylidc consensus tutorial figure
   (https://pylidc.github.io/tuts/consensus.html, LIDC-IDRI-0078, first nodule)
   from ``scan.to_volume()``: 4 reader contours + dashed 50 % consensus. Next to it
   is the same nodule from *our* preprocessed cube, so the two can be compared by eye.
2. For N random nodules, compares our resampled cubes with pylidc's native
   volume: consensus volume (mm^3) and mean HU inside the consensus mask. Big
   differences would mean misaligned images and masks.
3. Writes a grid of random preprocessed crops with reader and consensus contours.

    python -m lungseg.qa --dicom_root /kaggle/input --data /kaggle/working/lidc_crops --out docs/qa
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from lungseg.compat import import_pylidc
from lungseg.data.dicom_io import build_series_index

COLORS = ["r", "g", "b", "y"]


def _pylidc_volume(scan, series_dir):
    """pylidc's own loader, pointed at our DICOM folder."""
    import pydicom
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")]
    imgs = [pydicom.dcmread(f) for f in files]
    imgs = [im for im in imgs if hasattr(im, "ImagePositionPatient")]
    imgs.sort(key=lambda im: float(im.ImagePositionPatient[2]))
    zs, vol = [], []
    for im in imgs:  # drop duplicated z like we do
        z = float(im.ImagePositionPatient[2])
        if zs and abs(z - zs[-1]) < 1e-3:
            continue
        zs.append(z)
        vol.append(im.pixel_array * float(im.RescaleSlope) + float(im.RescaleIntercept))
    vol = np.stack(vol, axis=-1).astype(np.float32)  # (row, col, k) like pylidc
    if vol.shape[-1] != len(scan.slice_zvals):
        raise RuntimeError(f"{scan.patient_id}: {vol.shape[-1]} slices on disk vs {len(scan.slice_zvals)} in pylidc")
    return vol


def tutorial_figure(pl, index, crops_root, out_png, patient="LIDC-IDRI-0078"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pylidc.utils import consensus
    from skimage.measure import find_contours

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient).first()
    vol = _pylidc_volume(scan, index[scan.series_instance_uid])
    nods = scan.cluster_annotations(verbose=False)
    anns = nods[0]
    cmask, cbbox, masks = consensus(anns, clevel=0.5, pad=[(7, 7), (7, 7), (0, 0)])
    k = int(0.5 * (cbbox[2].stop - cbbox[2].start))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    ax = axes[0]
    ax.imshow(vol[cbbox][:, :, k], cmap=plt.cm.gray, alpha=0.5)
    for j in range(len(masks)):
        for c in find_contours(masks[j][:, :, k].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], COLORS[j], label=f"Annotation {j + 1}" if c is not None else None)
    for c in find_contours(cmask[:, :, k].astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], "--k", label="50% Consensus")
    ax.axis("off")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys())
    ax.set_title(f"pylidc reference path: {patient}, nodule 0, native slice {cbbox[2].start + k}")

    # our cube: same nodule (scan id, nodule 0), axial slice through the centre
    f = os.path.join(crops_root, "labeled", f"{patient}_s{scan.id:04d}_n00.npz")
    ax = axes[1]
    if os.path.exists(f):
        z = np.load(f)
        img, readers, mask = z["image"].astype(np.float32), z["readers"], z["mask"]
        # the axial plane in our cube nearest to the pylidc slice
        zc = float(scan.slice_zvals[cbbox[2].start + k])
        cz = [r for r in csv.DictReader(open(os.path.join(crops_root, "meta.csv"))) if r["file"].endswith(os.path.basename(f))]
        d = int(round((zc - float(cz[0]["center_z"])) / float(z["spacing"]) + (img.shape[0] - 1) / 2)) if cz else img.shape[0] // 2
        ax.imshow(np.clip(img[d], -1350, 150), cmap="gray", alpha=0.5)
        for j in range(readers.shape[0]):
            for c in find_contours(readers[j, d].astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], COLORS[j])
        for c in find_contours(mask[d].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "--k")
        ax.set_title(f"our preprocessed cube (0.8 mm iso), axial plane {d} (same z)")
    else:
        ax.text(0.5, 0.5, f"{f} not found (min_readers filter?)", ha="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def numeric_check(pl, index, crops_root, n=40, seed=0):
    """Consensus volume and mean HU: our cube vs pylidc native volume."""
    from pylidc.utils import consensus
    rows = list(csv.DictReader(open(os.path.join(crops_root, "meta.csv"))))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
    out = []
    by_scan = {}
    for i in pick:
        by_scan.setdefault(int(float(rows[i]["scan_id"])), []).append(rows[i])
    for sid, rs in by_scan.items():
        scan = pl.query(pl.Scan).filter(pl.Scan.id == sid).first()
        vol = _pylidc_volume(scan, index[scan.series_instance_uid])
        clusters = scan.cluster_annotations(verbose=False)
        vox_mm3 = float(scan.pixel_spacing) ** 2 * float(scan.slice_spacing)
        for r in rs:
            anns = clusters[int(float(r["nodule_idx"]))]
            cmask, cbbox, _ = consensus(anns, clevel=0.5)
            ref_vol = cmask.sum() * vox_mm3
            ref_hu = float(vol[cbbox][cmask].mean())
            z = np.load(os.path.join(crops_root, r["file"]))
            m = z["mask"].astype(bool)
            our_vol = m.sum() * float(z["spacing"]) ** 3
            our_hu = float(z["image"][m].mean()) if m.any() else float("nan")
            out.append(dict(file=r["file"], diameter_mm=float(r["diameter_mm"]), ref_vol_mm3=ref_vol,
                            our_vol_mm3=our_vol, vol_ratio=our_vol / max(ref_vol, 1e-6),
                            ref_mean_hu=ref_hu, our_mean_hu=our_hu, hu_diff=our_hu - ref_hu))
    return out


def crop_grid(crops_root, out_png, n=16, seed=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    rows = list(csv.DictReader(open(os.path.join(crops_root, "meta.csv"))))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
    cols = 8
    fig, axes = plt.subplots(int(np.ceil(len(pick) / cols)), cols, figsize=(2.2 * cols, 2.4 * np.ceil(len(pick) / cols)))
    for ax, i in zip(np.ravel(axes), pick):
        z = np.load(os.path.join(crops_root, rows[i]["file"]))
        img, mask, readers = z["image"], z["mask"], z["readers"]
        d = int(np.argmax(mask.sum((1, 2)))) if mask.any() else img.shape[0] // 2
        ax.imshow(np.clip(img[d], -1000, 400), cmap="gray")
        for j in range(readers.shape[0]):
            for c in find_contours(readers[j, d].astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], COLORS[j], lw=0.7)
        for c in find_contours(mask[d].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "--w", lw=1)
        ax.set_title(f"{rows[i]['patient_id'][-4:]} {float(rows[i]['diameter_mm']):.0f}mm "
                     f"R={rows[i]['n_readers'][:1]}", fontsize=7)
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.suptitle("random preprocessed nodules: reader contours (colours), 50% consensus (white dashed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom_root", required=True)
    ap.add_argument("--data", required=True, help="output of prepare_lidc")
    ap.add_argument("--out", default="docs/qa")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    pl = import_pylidc()
    index = build_series_index(args.dicom_root, os.path.join(args.data, "series_index.json"), verbose=False)
    try:
        tutorial_figure(pl, index, args.data, os.path.join(args.out, "pylidc_tutorial_0078.png"))
        print("wrote", os.path.join(args.out, "pylidc_tutorial_0078.png"))
    except Exception as e:
        print("tutorial figure skipped:", e)
    crop_grid(args.data, os.path.join(args.out, "crop_grid.png"))
    rows = numeric_check(pl, index, args.data, n=args.n)
    vr = np.array([r["vol_ratio"] for r in rows])
    hd = np.array([r["hu_diff"] for r in rows])
    summary = dict(n=len(rows), vol_ratio_median=float(np.median(vr)), vol_ratio_p5=float(np.percentile(vr, 5)),
                   vol_ratio_p95=float(np.percentile(vr, 95)), hu_diff_median=float(np.median(hd)),
                   hu_diff_abs_p95=float(np.percentile(np.abs(hd), 95)))
    json.dump(dict(summary=summary, rows=rows), open(os.path.join(args.out, "numeric_check.json"), "w"), indent=1)
    print("numeric check vs pylidc native volumes:", json.dumps(summary))


if __name__ == "__main__":
    main()
