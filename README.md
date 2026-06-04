<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px">Modeling Depth Ambiguity:<br>A Mixture-Density Representation for Flying-Point-Free Depth Estimation</h1>

**Siyuan Bian\*, Congrong Xu\*, Jun Gao**

<a href="https://biansy000.github.io/mda-site/"><img src="https://img.shields.io/badge/Project_Page-MDA-green" alt="Project Page"></a>
<a href="https://huggingface.co/sy000/MDA"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow" alt="Hugging Face Checkpoints"></a>
<a href="https://arxiv.org/abs/2606.02552"><img src="https://img.shields.io/badge/arXiv-2606.02552-b31b1b" alt="arXiv"></a>

</div>

This repository is the official code for **MDA**, from the paper *"Modeling Depth Ambiguity: A Mixture-Density Representation for Flying-Point-Free Depth Estimation"* ([arXiv](https://arxiv.org/abs/2606.02552) · [project page](https://biansy000.github.io/mda-site/)).

Common feed-forward depth models predict one depth value per pixel. At object edges this fails: the pixel covers both foreground and background, so its depth is *ambiguous*, and a single value falls between the two surfaces — a *flying point* that corrupts the reconstruction.

**MDA** replaces the single value with a *mixture density*: each pixel predicts a few depth hypotheses with probabilities, then picks one instead of averaging. This **largely eliminates flying points**, stays **robust to input blur**, adds **negligible overhead**, and works across backbones — both **DA3** and **VGGT**.

## 📰 News

- **2026-06-02:** Public release. Training code, evaluation scripts, and the `mda_mog_sky_l2` and `vggt_mog_l2` checkpoints are now available.



## 🚀 Quick Start

### 📦 Installation

MDA installs in two passes. The core package covers inference and the mixture-density head. A few extras are needed only for training, evaluation, or Gaussian-splatting export.

```bash
conda create -n mda python=3.10 -y
conda activate mda

# Install PyTorch for your CUDA version. This is one example.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Core package (mixture-density head + inference).
pip install -e .

# Optional: all inference extras (point-cloud viewers, format converters).
pip install -e ".[all]"
```

**Training and evaluation extras.** The Hydra/Lightning training launcher and the benchmark eval scripts use a few libraries that `pyproject.toml` does not declare. Install them when you need to train or evaluate:

```bash
# Training stack.
pip install hydra-core lightning lightning-bolts torchmetrics rootutils \
            accelerate peft pyyaml tqdm

# Eval and visualization utilities (src/testing/eval_cut3r/*, src/training/da3_wrapper.py).
pip install matplotlib scipy sympy
```

**ffmpeg.** The `bash src/testing/run_demo.sh <video>` flow extracts frames from a video, so `ffmpeg` must be on your `PATH`:

- Debian / Ubuntu: `sudo apt-get install ffmpeg`
- macOS (Homebrew): `brew install ffmpeg`

### 🧱 Download the checkpoints

The pretrained MDA checkpoints are on the Hugging Face Hub at [`sy000/MDA`](https://huggingface.co/sy000/MDA). Download them into `checkpoints/MDA/`, the path `src/testing/utils/model_choice.py` expects:

```bash
hf download sy000/MDA --local-dir checkpoints/MDA
```

This places two checkpoints:

| `--model_name` | Backbone | Checkpoint file | Notes |
|---|---|---|---|
| `mda_mog_sky_l2` **(default)** | DA3 Giant + Gaussian mixture + sky | `checkpoints/MDA/DA3_MOG_Sky_LogL2.ckpt` | Default model; main results in the paper. |
| `vggt_mog_l2` | VGGT-1B + MDA head | `checkpoints/MDA/VGGT_MOG_LogL2.ckpt` | Same head on a VGGT-1B backbone. |

(`hf` ships with `huggingface_hub`. Run `pip install -U huggingface_hub` if the command is missing.)

### 💻 Run the demo

`src/testing/run_demo.sh` wraps `src/testing/run_inference_video.py`. It takes either a folder of images or a single video file. For a video, it extracts frames with `ffmpeg` first.

```bash
# 1. Video file (frames extracted to a temp dir, removed on exit).
bash src/testing/run_demo.sh assets/examples/parkour_video.mp4 \
    --output_dir eval_results/demo/parkour --fps 5

# 2. Folder of images.
bash src/testing/run_demo.sh path/to/image_folder

# 3. Use the VGGT model instead of the default DA3 model.
bash src/testing/run_demo.sh assets/examples/parkour_video.mp4 \
    --model_name vggt_mog_l2 \
    --output_dir eval_results/demo/my_scene
```

The default model is `mda_mog_sky_l2`. Override it with `--model_name` (see the table above, or `src/testing/utils/model_choice.py` for all names). Outputs go to `--output_dir`, which defaults to `eval_results/demo/<input_basename>/`: per-frame depth maps, an optional side-view point cloud `.ply`, and per-frame `cameras.npz`.

## 🏋️ Training

Training uses Hydra to compose an experiment config under `configs/experiment/mda/` (the `.yaml` extension is implicit). Each config finetunes a pretrained DA3 or VGGT checkpoint with **K = 4** mixture components for **10k steps** on **4 × RTX Pro 6000**, learning rate **1e-4**, batch size **48** (paper §5.1.1).

```bash
# Default: DA3 + Gaussian mixture + sky component.
python src/training/train.py experiment=mda/da3_mog_sky_full
```

Other recipes under `configs/experiment/mda/`:

| Config | Description |
|---|---|
| `da3_mog_sky_full` | DA3 + Gaussian mixture + sky component **(default)** |
| `da3_mog_sky_full_l1` | DA3 + Laplacian mixture (paper Table 1, "LMM" row) |
| `vggt_mog_full` | VGGT backbone + MDA head |

Override any Hydra field on the command line:

```bash
python src/training/train.py experiment=mda/da3_mog_sky_full \
    trainer.devices=4 data.num_views=8 logger=wandb
```

**Training data.** The synthetic training mix follows the DA3 recipe: AriaSyntheticENV, HyperSim, MvsSynth, OmniWorld, PointOdyssey, TartanAir, vKitti2, DynamicReplica, UnrealStereo4K (paper §5.1.1).

## 📊 Evaluation

Two launcher scripts cover the two benchmark tracks in the paper. They use the same checkpoints as the demo. Each script selects the model by name through `src/testing/utils/model_choice.py`, and both default to `mda_mog_sky_l2`. To evaluate a different model, edit the `model_names` array at the top of the script (for example, set it to `vggt_mog_l2`).

```bash
# Boundary-quality benchmark (NRGBD, 7Scenes, HiRoom) — paper Table 1.
bash src/testing/eval_cut3r/mv_recon/run_mv_recon.sh

# Video-depth benchmark (Sintel, Bonn, KITTI, DIODE) — paper Table 2.
bash src/testing/eval_cut3r/video_depth/run_video_depth.sh
```

Both scripts write per-dataset and per-model outputs under `eval_results/`. 
## 📝 Citation

If you build on **MDA**, please cite:

```bibtex
@misc{bian2026modeling,
  title         = {Modeling Depth Ambiguity: A Mixture-Density Representation for Flying-Point-Free Depth Estimation},
  author        = {Siyuan Bian and Congrong Xu and Jun Gao},
  year          = {2026},
  eprint        = {2606.02552},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.02552}
}
```

## 🙏 Acknowledgements

This codebase builds on these open-source releases:

- [**Depth Anything 3**](https://github.com/ByteDance-Seed/Depth-Anything-3) — one of the two backbones, and the source of the DINOv2-based encoder, DPT head, and inference code.
- [**Stream3R**](https://github.com/NIRVANALAN/STream3R) — the Hydra/Lightning training launcher, multi-view DUSt3R data modules, and streaming VGGT-style sequence wrapper.

We thank the authors for their work.
