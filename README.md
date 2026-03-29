# 3D Vision Mamba for Lung Nodule Segmentation

A volumetric segmentation system for the LIDC-IDRI dataset using a state-of-the-art 3D Vision Mamba architecture.

## Project Structure

- `data/`: Contains preprocessing logic for DICOM volumes, Hounsfield Unit (HU) normalization, resampling, and 3D patch extraction.
- `models/`: Implementation of the 3D Vision Mamba architecture, including volumetric patch embedding, S6-based SSM blocks, and the 3D decoder.
- `training/`: Training workflows, custom Dice-BCE loss functions, and optimization routines.
- `evaluation/`: Scripts to calculate volumetric metrics such as Dice coefficient, IoU, Sensitivity, and Precision.
- `inference/`: Prediction pipeline for full volumes and slice-overlay visualization.
- `docs/`: Detailed system documentation and analysis.

## Core Concepts

### Lung Nodules
Lung nodules are small masses of tissue in the lungs. Early detection and segmentation on CT scans are critical for lung cancer staging and treatment planning.

### Volumetric Processing
Conventional systems often process CT scans as independent 2D slices. This system treats the CT scan as a true 3D volume, preserving 3D morphology and voxel-to-voxel relationships across all axes.

### Vision Mamba (SSM)
State Space Models (SSM), specifically the Mamba architecture, allow for long-range dependency modeling with linear complexity. In 3D medical imaging, where volumes contain millions of voxels (tokens), this efficiency is vital compared to the quadratic cost of Vision Transformers.

## Getting Started

### 1. Prerequisites
Ensure you have the required libraries installed via `requirements.txt`. The pipeline leverages the LIDC-IDRI dataset (provided in your manifest folder). During execution, the codebase natively configures `pylidc` environments automatically.

### 2. Training
Trigger the training sequence of the Vision Mamba Diffusion architecture using:
```bash
python main.py --train --manifest_dir "manifest-1585167679499" --epochs 10 --batch_size 2
```

### 3. Inference / Generation
Generate denoised nodule masks conditionally via the pure Vision Mamba backbone using:
```bash
python main.py --predict --manifest_dir "manifest-1585167679499"
```
Predictions and baseline CT slices will be output visually to `docs/sample_prediction.png`.

## Requirements
- PyTorch
- NumPy
- pydicom
- scipy
- matplotlib
- scikit-image
