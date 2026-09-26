# Label-efficient lung nodule segmentation with a 3D Mamba-UNet and diffusion pretraining

3D segmentation of lung nodules on **LIDC-IDRI**, built to work well when only a
small number of scans have labels. The pieces:

* **3D Mamba-UNet.** Residual conv encoder/decoder, with tri-directional Mamba (S6
  selective state-space) layers at the 16³, 8³ and 4³ stages. The Mamba layers give
  global context in linear time.
* **Diffusion-denoising pretraining (DDeP).** The whole network, decoder included,
  is first trained as a DDPM-style noise predictor on *unlabelled* CT cubes, then
  fine-tuned on the few labelled ones. This is where the diffusion idea of the
  original proposal is kept: it helps when labels are scarce and adds no inference cost.
* **Label-efficiency protocol.** Train on 10 / 25 / 50 / 100% of the labelled
  training *patients* (nested subsets), with equal compute per run, and test every
  model on the same held-out patients.
* **Human reference.** Every test nodule also gets a leave-one-radiologist-out
  Dice/IoU, so the model is compared against how much the four LIDC radiologists
  agree with each other.

> **Status:** the code is complete and tested (unit tests plus an end-to-end
> synthetic run), but **no LIDC numbers are reported yet**: they come from
> running the Kaggle notebook. Nothing in this README is a made-up result.
> Fill in the table below from `runs/summary.md`.

## Results

| config | 10% labels | 25% | 50% | 100% |
|---|---|---|---|---|
| Residual 3D UNet (scratch) | – | – | – | – |
| Mamba-UNet (scratch) | – | – | – | – |
| **Mamba-UNet + diffusion pretraining** | – | – | – | – |
| Radiologists (leave-one-out) | – | | | |

Metric: per-nodule 3D Dice against the 50% consensus mask (test patients only),
flip-TTA, and central connected component kept. IoU, precision, recall, HD95 and
per-size breakdowns are in `runs/*/results.json`.

**What to expect.** Crop-based methods in the LIDC literature mostly report
Dice ≈ 0.80–0.87 on consensus masks, and the four LIDC radiologists agree with
each other at about the same level. **IoU is always lower than Dice for the same
masks** (IoU = D / (2 − D)): Dice 0.85 is IoU 0.74, and IoU 0.80 needs Dice 0.89.
So Dice > 0.8 is a realistic target here, while IoU > 0.8 would be above what
LIDC's own annotators achieve between themselves. Papers with much higher
numbers usually evaluate on 2D slices, pick easy nodules, or split by slice
instead of by patient (which leaks data).

## Data protocol

* Source: all 1,018 LIDC-IDRI CT scans with pylidc's annotation database. Nodules
  are ≥3 mm by LIDC definition, and only nodules marked by **≥2 radiologists** are
  kept: 1,880 nodules from 796 patients.
* Each nodule is stored as a 96³ cube at 0.8 mm isotropic spacing, centred on the
  50% consensus. Each reader's own mask is kept too. Training samples random 64³
  views (rotation, tilt, 0.8–1.25 scale, flips, ±6 mm off-centre, intensity,
  blur, and thick-slice simulation), all done on the GPU.
* Splits are by **patient**: 70% train, 10% val, 20% test, seed 0. Label fractions
  subsample the *training* patients only, and the subsets are nested
  (10% ⊂ 25% ⊂ 50% ⊂ 100%).
* Pretraining uses images only: all training patients plus random lung cubes from
  patients without qualifying nodules. **Val/test patients are never used.**
* **Preprocessing QA** (`python -m lungseg.qa`). It re-creates the official
  [pylidc consensus tutorial](https://pylidc.github.io/tuts/consensus.html) figure
  (LIDC-IDRI-0078, 4 readers + 50% consensus) from the raw DICOM, puts our
  resampled cube of the same nodule next to it, and writes a grid of random
  crops. For random nodules it also checks the consensus volume and mean HU
  against pylidc's native-resolution volume. The consensus volume should match
  to within a few percent. The mean HU inside the mask comes out a little lower
  because of partial-volume smoothing at the boundary.
* Task definition: segmentation of a *given* nodule (the crop centre is known).
  This is the standard LIDC segmentation benchmark setting. It is not detection.

## Quick start (Kaggle, free GPU)

Open `notebooks/kaggle_lidc_mamba.ipynb` on Kaggle. Turn on GPU and internet, add
the dataset `washingtongold/lidcidri30` (original TCIA DICOM folders for about a third of
LIDC-IDRI, 41 GB), and *Run all*. `justinkirby/the-cancer-imaging-archive-lidcidri`
contains only the XML annotations, no images. By hand:

```bash
pip install -r requirements.txt
# optional, faster Mamba kernels on NVIDIA GPUs
pip install causal-conv1d mamba-ssm --no-build-isolation

python -m pytest -q tests                                       # sanity checks
python -m lungseg.data.prepare_lidc --dicom_root /path/to/LIDC-IDRI --out data/lidc_crops
python -m lungseg.experiments --data data/lidc_crops --runs runs  # whole study, resumable
python -m lungseg.evaluate  --data data/lidc_crops --ckpt runs/mamba_ddep_f1_s0/best.pt
python -m lungseg.visualize --data data/lidc_crops --ckpt runs/mamba_ddep_f1_s0/best.pt
```

Single runs:

```bash
python -m lungseg.pretrain --data data/lidc_crops --out runs/pretrain_mamba --arch mamba
python -m lungseg.train    --data data/lidc_crops --out runs/my_run --arch mamba --fraction 0.1 \
                           --init runs/pretrain_mamba/pretrain.pt
```

No DICOM yet? `python -m lungseg.data.download_tcia --out LIDC-IDRI --n_patients 300`
pulls the annotated CT series straight from TCIA's public API. Try it on
Colab: it could not be tested from the machine this was written on.

CPU-only smoke test with synthetic data (no LIDC needed):

```bash
python -m lungseg.data.synthetic --out data/synthetic --n_patients 30
python -m lungseg.train --data data/synthetic --out runs/syn --steps 300 --crop 48 --widths 16 32 64 64 96
```

## Repository layout

```
lungseg/
  compat.py               numpy/pkg_resources shims so pylidc works on modern numpy
  data/prepare_lidc.py    DICOM + pylidc -> nodule cubes, reader masks, patient splits
  data/dicom_io.py        series index + lazy per-slice decoding (reads only the slices it needs)
  data/dataset.py         in-RAM crop sets, nested label fractions, GPU augmentation
  data/download_tcia.py   optional downloader for the TCIA public API
  data/synthetic.py       synthetic cache with the same format, for tests
  models/ssm.py           Mamba/S6: mamba_ssm CUDA | PyTorch parallel scan | reference scan
  models/mamba_unet.py    3D Mamba-UNet and the matched residual-UNet baseline
  pretrain.py             diffusion-denoising pretraining
  train.py                supervised fine-tuning (Dice+BCE, deep supervision, EMA, AMP)
  evaluate.py             per-nodule Dice/IoU/precision/recall/HD95 + radiologist agreement
  experiments.py          label-efficiency grid + summary table and plot
  visualize.py            qualitative figure
notebooks/kaggle_lidc_mamba.ipynb
tests/                    scan maths, causality, model shapes, metrics, preprocessing alignment
docs/original_proposal.pdf
```

## Why these design choices (limited labels)

* **Conv stages at high resolution, Mamba at low resolution.** Conv layers are
  sample-efficient for edges and texture. Global mixing matters at coarse scales
  (telling a nodule from a vessel it touches, or from the pleura). A pure
  patch-16 ViM, as in the first version, loses the 3–10 mm nodules entirely.
* **Denoising pretraining over contrastive or MAE pretraining.** It trains
  encoder *and* decoder on unlabelled data, and the decoder is most of a
  segmentation network. It is also the part of the original diffusion idea that
  pays off for segmentation.
* **Nodule-centred 3D crops, isotropic resampling.** These remove the class
  imbalance and the slice-thickness variation (0.6–3 mm) that dominate
  full-volume training on small data.
* **EMA weights, heavy augmentation done on the GPU, fixed step budget,
  consensus-of-readers targets.** Standard ways to keep variance low when there
  are only ~100 labelled nodules.
* **A matched baseline.** `--arch unet` swaps every Mamba layer for a residual
  conv block. If the Mamba layers don't help, the table will show it.

## What changed from the first version

The first version could not produce a meaningful result:

* the dataset returned all-zero masks;
* when `mamba-ssm` was missing, the "Mamba" layer silently became a single `nn.Linear`;
* the demo painted a synthetic nodule onto the CT and trained on that single image;
* the committed `.pth` had been trained on those empty masks.

All of that has been replaced. The proposal PDF is kept in `docs/`.

## Limitations

* One patient-level split. Use `--seeds 0 1 2` in `experiments.py` for variance
  over training seeds; cross-validation is not implemented.
* The nodule position is assumed known (a crop around it). A detector
  would be needed for a fully automatic pipeline.
* LIDC masks carry real inter-reader variability. Scores well above the
  radiologist leave-one-out agreement should be viewed with suspicion.

Authors of the original proposal: Uday Yerraballi, Chris Jason Baskar.
