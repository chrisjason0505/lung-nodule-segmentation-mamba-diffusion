"""Fast DICOM access for LIDC-IDRI.

pylidc's ``scan.to_volume()`` decodes every slice of a series. We only need the
slices around each nodule, so we read headers first (cheap, no pixel data) and
decode pixels lazily for the slices we actually use.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pydicom


def _is_dicom_name(name: str) -> bool:
    return name.lower().endswith(".dcm")


def build_series_index(root: str, cache_path: str | None = None, verbose: bool = True) -> dict:
    """Map SeriesInstanceUID -> directory for every DICOM series under ``root``.

    Works with any TCIA layout (``LIDC-IDRI/LIDC-IDRI-0001/<study>/<series>/*.dcm``,
    the Kaggle mirror, NBIA downloads...). Only one header per directory is read.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            idx = json.load(f)
        if idx.get("_root") == os.path.abspath(root):
            return idx["series"]
    series = {}
    n_dirs = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        dcms = sorted(f for f in filenames if _is_dicom_name(f))
        if not dcms:
            continue
        n_dirs += 1
        for name in dcms[:3]:  # first readable header wins
            try:
                ds = pydicom.dcmread(os.path.join(dirpath, name), stop_before_pixels=True,
                                     specific_tags=["SeriesInstanceUID", "Modality", "PatientID"])
                uid = str(ds.SeriesInstanceUID)
                if str(getattr(ds, "Modality", "CT")) == "CT":
                    series[uid] = dirpath
                break
            except Exception:
                continue
    if verbose:
        print(f"[index] {len(series)} CT series found in {n_dirs} DICOM directories under {root}")
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"_root": os.path.abspath(root), "series": series}, f)
    return series


@dataclass
class LazySeries:
    """Header-only view of a CT series with on-demand slice decoding (HU, float32)."""

    directory: str
    files: list = field(default_factory=list)
    z: np.ndarray = None           # sorted ascending, one per unique slice
    rows: int = 0
    cols: int = 0
    _cache: dict = field(default_factory=dict)

    @classmethod
    def open(cls, directory: str) -> "LazySeries":
        entries = []
        rows = cols = 0
        for name in os.listdir(directory):
            if not _is_dicom_name(name):
                continue
            path = os.path.join(directory, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True,
                                     specific_tags=["ImagePositionPatient", "Rows", "Columns",
                                                    "InstanceNumber"])
                z = float(ds.ImagePositionPatient[2])
            except Exception:
                continue
            rows, cols = int(ds.Rows), int(ds.Columns)
            entries.append((z, path))
        if not entries:
            raise FileNotFoundError(f"no readable CT slices in {directory}")
        entries.sort(key=lambda e: e[0])
        # drop duplicated z positions (a few LIDC series contain them)
        zs, files = [], []
        for z, p in entries:
            if zs and abs(z - zs[-1]) < 1e-3:
                continue
            zs.append(z)
            files.append(p)
        return cls(directory=directory, files=files, z=np.asarray(zs, np.float64), rows=rows, cols=cols)

    def slice_hu(self, index: int) -> np.ndarray:
        if index not in self._cache:
            ds = pydicom.dcmread(self.files[index])
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
            self._cache[index] = arr * slope + intercept
        return self._cache[index]

    def index_of_z(self, zval: float, tol: float = 0.51) -> int:
        k = int(np.argmin(np.abs(self.z - zval)))
        if abs(self.z[k] - zval) > tol:
            raise KeyError(f"no slice at z={zval:.2f} (closest {self.z[k]:.2f}) in {self.directory}")
        return k

    def clear(self):
        self._cache.clear()
