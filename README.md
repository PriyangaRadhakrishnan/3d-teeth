# Teeth3DS-Based Sparse-View Dental Reconstruction Prototype

This repository currently implements **Phase 1-5 only**: OBJ inspection, non-destructive working-copy normalization, five controlled Blender views, ground-truth RGB/depth/normal/mask artifacts, camera metadata, and contact-sheet validation.

The experiment is synthetic and controlled: images, geometry, and camera poses are generated from the original OBJ. It must not be described as a clinical DenGaussDiff reproduction.

## Upstream inspection

- [IOSSAM](https://github.com/ar-inspire/IOSSAM): its README documents `prepare_3d_data.py`, `prepare_2d_data.py`, and `prepare_2d_GT.py`, but the current `master` tree contains only `README.md` and `requirements.txt`; therefore no preprocessing source was copied. We reuse the documented Teeth3DS-to-Blender workflow and view-generation idea.
- [Teeth3DS repository](https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges): annotations contain per-vertex `labels` and `instances`; the loader checks their lengths against the OBJ vertex count. Dataset license: CC BY-NC-ND 4.0.
- [gsplat](https://github.com/nerfstudio-project/gsplat) and [official Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): reserved for Phase 6+ after rendered artifacts validate.
- [ControlNet++](https://github.com/liming-ai/ControlNet_Plus_Plus): optional later research adapter; no model or training data is included here.

## Environment

Use a fresh Python 3.10 or 3.11 environment. Install `requirements.txt`, then install Blender 4.x separately and ensure `blender` is on `PATH`. The old IOSSAM pins (`torch==1.13`, `open3d==0.16`, two incompatible OpenCV pins) are intentionally not installed in this Phase 1-5 environment. Gaussian and diffusion dependencies should live in separate environments because their CUDA/PyTorch requirements are likely to conflict.

## Commands

Keep the original dataset under `data/raw/Teeth3DS`; do not modify it.

```powershell
python -m pip install -r requirements.txt
python scripts/inspect_obj.py data/raw/Teeth3DS/<sample>.obj --json data/raw/Teeth3DS/<sample>.json
python scripts/render_dataset.py data/raw/Teeth3DS/<sample>.obj --blender blender
python scripts/validate_sample.py data/processed/<sample>
```

The output sample contains `images/*.png`, metric camera-Z `depth/*.exr`, camera-space encoded `normals/*.png`, binary `masks/*.png`, `cameras/cameras.json`, `mesh_transform.json`, `inspection.json`, and report contact sheets. `mesh_transform.json` records the normalized working-copy transform needed to return a reconstruction to source coordinates.

## Reconstruction baseline

The current controlled baseline fuses masked depth pixels from all five calibrated views into a 3D PLY:

```powershell
python scripts/reconstruct.py data/processed/0EJBIPTC_lower --blender "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"
python scripts/register.py data/processed/0EJBIPTC_lower/reconstruction.ply data/raw/Teeth3DS/0EJBIPTC_lower.obj --transform data/processed/0EJBIPTC_lower/mesh_transform.json
```

This produces `reconstruction.ply`, `registered_reconstruction.ply`, and `registration_result.json`. On the current synthetic sample, the prototype achieved fitness `1.0` and inlier RMSE approximately `0.349` in the source mesh units. This is a depth-fusion/ICP baseline, not a trained Gaussian field or DenGaussDiff reproduction.

## Gaussian baseline

The first trainable experiment is a fixed-geometry `gsplat` baseline. It optimizes Gaussian color, opacity, and scale from the five RGB views while initializing positions from the depth-fused PLY:

```powershell
python scripts/train_gaussian.py data/processed/0EJBIPTC_lower --iterations 300 --max-points 12000
```

PyTorch CUDA and `gsplat` are installed, but `gsplat` also requires the NVIDIA CUDA Toolkit compiler (`nvcc`) to build its native backend. The current machine has the CUDA runtime and RTX 4060 but does not yet have `nvcc`; install the CUDA Toolkit before running this trainer.

Blender must be installed before the render command can run. No rendered output is claimed in this environment because Blender and a Teeth3DS OBJ are not currently present.