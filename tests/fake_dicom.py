"""Write a fake DICOM series for a real pylidc scan: -850 HU everywhere except
the voxels that the radiologists annotated (set to +40 HU). Lets us check that
prepare_lidc puts masks and images in the same place without the 125 GB dataset."""
import os

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from lungseg.compat import import_pylidc


def write_fake_series(patient_id, out_root, shuffle_files=True):
    pl = import_pylidc()
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    zs = np.asarray(scan.slice_zvals)
    n = len(zs)
    vol = np.full((512, 512, n), -850, np.int16)
    for a in scan.annotations:
        m = a.boolean_mask()
        si, sj, sk = a.bbox()
        sub = vol[si, sj, sk]
        sub[m] = 40
        vol[si, sj, sk] = sub
    d = os.path.join(out_root, "LIDC-IDRI", patient_id, "study", "series")
    os.makedirs(d, exist_ok=True)
    order = np.random.default_rng(0).permutation(n) if shuffle_files else np.arange(n)
    for fi, k in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.PatientID = patient_id
        ds.Modality = "CT"
        ds.SeriesInstanceUID = scan.series_instance_uid
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = int(k) + 1
        ds.ImagePositionPatient = [-150.0, -150.0, float(zs[k])]
        ds.PixelSpacing = [float(scan.pixel_spacing)] * 2
        ds.Rows, ds.Columns = 512, 512
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 0
        ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
        ds.PixelData = (vol[:, :, k].astype(np.int32) + 1024).astype(np.uint16).tobytes()
        ds.save_as(os.path.join(d, f"{fi:06d}.dcm"), enforce_file_format=True)
    return scan
