# Zero-DCE from Scratch (Assignment Reimplementation)

This repository contains a **from-scratch PyTorch reimplementation** of **Zero-DCE: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement**.

The paper reformulates low-light enhancement as **iterative curve estimation** with a lightweight **DCE-Net**, and trains the model **without paired or unpaired reference images** using four non-reference losses: **spatial consistency**, **exposure control**, **color constancy**, and **illumination smoothness**. The paper reports strong quantitative performance and evaluates full-reference quality with **PSNR**, **SSIM**, and **MAE** on the paired SICE Part2 subset. In Table 2, Zero-DCE reports **16.57 PSNR**, **0.59 SSIM**, and **98.78 MAE**. 
Paper link: https://arxiv.org/pdf/2001.06826
Official Github Repo link: https://github.com/Li-Chongyi/Zero-DCE
Blog link: https://medium.com/@jatsyajariwala29/zero-dce-a-simple-yet-powerful-approach-for-low-light-image-enhancement-f887c7e7348d 

## What this repo includes

- `zerodce/model.py` — DCE-Net and iterative curve application
- `zerodce/losses.py` — Zero-DCE loss functions
- `zerodce/dataset.py` — recursive image loader with augmentation
- `zerodce/train.py` — training from scratch
- `zerodce/enhance.py` — inference on a folder of images
- `zerodce/compare_outputs.py` — compare our outputs with official outputs and original inputs
- `tools/make_toy_subset.py` — create a balanced toy subset for fast experiments
- `lowlight_train.py`, `lowlight_test.py`, `Myloss.py`, `model.py`, `dataloader.py` — official code files kept for reference and getting output of official model for test data to compare with our model

## Project goal

The goal of the assignment is to:

1. Reimplement the paper from scratch.
2. Train on a **toy dataset / small sample**.
3. Compare the reimplementation against the **official implementation**.
4. Report both **quantitative** and **qualitative** differences.

## Folder layout

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── train_data/
│   ├── toy_train_data/
│   └── test_data/
├── outputs/
│   ├── ours/
│   └── official/
├── runs/
├── snapshots/                # official model checkpoints
├── comparison_report/
├── tools/
│   └── make_toy_subset.py
├── zerodce/
│   ├── model.py
│   ├── losses.py
│   ├── dataset.py
│   ├── train.py
│   ├── enhance.py
│   └── compare_outputs.py
├── lowlight_train.py        # official training script
├── lowlight_test.py         # official inference script
├── Myloss.py
├── model.py
└── dataloader.py
```

## Dataset setup

Place the downloaded training images in:

```text
data/train_data/
```

The loader reads images recursively, so subfolders are supported.

For a quick experiment, create a balanced toy subset:

```bash
python tools/make_toy_subset.py --src data/train_data --dst data/toy_train_data --max_images 256 --bins 4
```

## Install

```bash
pip install -r requirements.txt
```

## Sequential run order

### 1) Create a toy subset

```bash
python tools/make_toy_subset.py --src data/train_data --dst data/toy_train_data --max_images 256 --bins 4
```

### 2) Train the from-scratch model

Recommended 5-epoch demo run:

```bash
python -m zerodce.train \  
  --data_dir data/toy_train_data \
  --epochs 5 \  
  --batch_size 8 \  
  --tv_weight 200 \  
  --exp_weight 8 \  
  --color_weight 5 \  
  --spa_weight 1 \  
  --save_dir runs/toy_better
```

For slightly stronger enhancement in the toy setting, the default training script uses a modest exposure emphasis and stronger smoothness regularization to avoid over-brightening.

### 3) Run inference with your checkpoint

```bash
python -m zerodce.enhance \
  --checkpoint runs/toy_better/checkpoints/zerodce_epoch_005.pt \
  --input_dir data/test_data \
  --output_dir outputs/ours
```

### 4) Run the official implementation on the same test images

Save the official outputs to:

```text
outputs/official/
```

### 5) Compare outputs

```bash
python -m zerodce.compare_outputs \
  --ours outputs/ours \
  --official outputs/official \
  --original data/test_data \
  --report_dir comparison_report \
  --save_panels
```

This produces:

- `comparison_report/per_image.csv`
- `comparison_report/summary.json`
- `comparison_report/report.html`
- `comparison_report/visuals/`

## What the comparison means

The comparison script reports three kinds of values:

- **ours vs official** — how closely our implementation matches the official repo
- **ours vs original** — how much our model changes the input image
- **official vs original** — how much the official model changes the input image

For a fair reproduction study, the most useful report is the **qualitative panel** showing:

- **ORIGINAL**
- **OFFICIAL**
- **OURS**

and the CSV/JSON summary for the corresponding metrics.

## Expected result pattern

With only a 5-epoch toy run, the output is expected to differ from the official model and may be brighter or less stable than the official result. That is normal for a quick reproduction experiment. The main objective is to show that the model is functioning, the losses decrease overall, and the enhancement trend matches the paper.

## Citation

Paper:

> Chunle Guo et al., *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*, CVPR 2020.

