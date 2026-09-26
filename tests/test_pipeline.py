import json
import os

import numpy as np
import pytest
import torch

from lungseg.metrics import binary_metrics, keep_central_component, reader_agreement
from lungseg.models.mamba_unet import MambaLayer3D, build_model
from lungseg.models.ssm import Mamba, selective_scan_parallel, selective_scan_ref


def test_parallel_scan_matches_reference():
    torch.manual_seed(0)
    for L in (16, 37, 64):
        b, d, n = 2, 6, 4
        u = torch.randn(b, d, L, dtype=torch.float64, requires_grad=True)
        delta = torch.rand(b, d, L, dtype=torch.float64)
        A = -torch.rand(d, n, dtype=torch.float64) * 4
        B, C = torch.randn(b, n, L, dtype=torch.float64), torch.randn(b, n, L, dtype=torch.float64)
        D = torch.randn(d, dtype=torch.float64)
        y1, y2 = selective_scan_ref(u, delta, A, B, C, D), selective_scan_parallel(u, delta, A, B, C, D)
        assert torch.allclose(y1, y2, atol=1e-10)
        g1, = torch.autograd.grad(y1.pow(2).sum(), u)
        g2, = torch.autograd.grad(y2.pow(2).sum(), u)
        assert torch.allclose(g1, g2, atol=1e-9)


def test_mamba_is_causal_and_selective():
    torch.manual_seed(0)
    m = Mamba(16).double()
    x = torch.randn(1, 32, 16, dtype=torch.float64)
    y = m(x)
    x2 = x.clone()
    x2[:, 20:] += 1.0  # change the future
    y2 = m(x2)
    assert torch.allclose(y[:, :20], y2[:, :20])        # causal
    assert not torch.allclose(y[:, 20:], y2[:, 20:])


def test_mamba_param_names_match_mamba_ssm():
    names = set(Mamba(32).state_dict().keys())
    assert names == {"A_log", "D", "in_proj.weight", "conv1d.weight", "conv1d.bias", "x_proj.weight",
                     "dt_proj.weight", "dt_proj.bias", "out_proj.weight"}


def test_mamba_layer_global_receptive_field():
    """A change in one corner voxel must influence the opposite corner (global mixing)."""
    torch.manual_seed(0)
    layer = MambaLayer3D(8).double().eval()
    x = torch.randn(1, 8, 4, 4, 4, dtype=torch.float64)
    x2 = x.clone()
    x2[..., -1, -1, -1] += torch.randn(8, dtype=torch.float64)  # (a constant shift would be removed by LayerNorm)
    diff = (layer(x) - layer(x2))[..., 0, 0, 0].abs().max().item()
    assert diff > 1e-9  # small at init (SSM decay), but nonzero: the receptive field is global


@pytest.mark.parametrize("arch", ["mamba", "unet"])
def test_model_shapes(arch):
    m = build_model(arch, widths=(8, 16, 16, 32, 32))
    x = torch.randn(2, 1, 32, 32, 32)
    out, ds = m.train()(x)
    assert out.shape == x.shape and [d.shape[-1] for d in ds] == [16, 8]
    assert m.eval()(x).shape == x.shape


def test_metrics_and_postprocess():
    a = np.zeros((20, 20, 20), np.uint8)
    a[8:12, 8:12, 8:12] = 1
    m = binary_metrics(a, a, 0.8)
    assert m["dice"] == 1 and m["iou"] == 1 and m["hd95_mm"] == 0
    b = a.copy()
    b[0:2, 0:2, 0:2] = 1  # far away false positive
    assert (keep_central_component(b) == a).all()
    readers = np.stack([a, a, a])
    assert reader_agreement(readers)["dice"] == 1


def test_prepare_lidc_alignment(tmp_path):
    """Real pylidc annotations + fake DICOM: after preprocessing the bright voxels
    of the image must coincide with the stored reader masks."""
    pytest.importorskip("pylidc")
    from tests.fake_dicom import write_fake_series
    from lungseg.data.prepare_lidc import main as prepare

    write_fake_series("LIDC-IDRI-0078", str(tmp_path / "dicom"))
    out = tmp_path / "crops"
    prepare(["--dicom_root", str(tmp_path / "dicom"), "--out", str(out), "--workers", "1",
             "--unlabeled_per_scan", "1", "--min_readers", "1"])
    import csv
    rows = list(csv.DictReader(open(out / "meta.csv")))
    assert len(rows) == 4  # LIDC-IDRI-0078 has 4 annotated nodules
    for r in rows:
        z = np.load(out / r["file"])
        img, mask, readers = z["image"], z["mask"], z["readers"]
        assert img.shape == (96, 96, 96) and readers.shape[0] == int(r["n_readers"])
        bright = img > -400
        union = readers.max(0) > 0
        d = 2 * (bright & union).sum() / (bright.sum() + union.sum())
        assert d > 0.85, f"image/mask misaligned in {r['file']}: dice {d:.3f}"
        # consensus centred in the cube
        com = np.argwhere(mask).mean(0)
        assert np.all(np.abs(com - 47.5) < 3)
    splits = json.load(open(out / "splits.json"))
    assert sum(len(splits[k]) for k in ("train", "val", "test")) == 1
    assert len(os.listdir(out / "unlabeled")) == 1


def test_checkpointing_gives_identical_gradients():
    torch.manual_seed(0)
    m = build_model("mamba", widths=(8, 16, 16, 32, 32))
    x = torch.randn(1, 1, 32, 32, 32)

    def grads():
        m.zero_grad()
        out, ds = m.train()(x)
        (out.mean() + sum(d.mean() for d in ds)).backward()
        return [p.grad.clone() for p in m.parameters()]

    layers = [l for l in m.modules() if isinstance(l, MambaLayer3D)]
    assert all(l.use_checkpoint for l in layers)  # auto-on for the PyTorch scan
    g1 = grads()
    for l in layers:
        l.use_checkpoint = False
    g2 = grads()
    assert all(torch.allclose(a, b) for a, b in zip(g1, g2))
