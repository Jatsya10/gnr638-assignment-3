# Zero-DCE from scratch

This project is a clean PyTorch reimplementation of **Zero-DCE** for low-light image enhancement.

The paper frames enhancement as **iterative curve estimation** with a lightweight DCE-Net, and trains without paired ground truth using four non-reference losses: spatial consistency, exposure control, color constancy, and illumination smoothness. The official repo uses a 7-layer convolutional network with symmetric concatenation and 24 output channels for 8 curve iterations. citeturn301122view3turn770689view0turn770689view1turn770689view2turn770689view3turn808003view2

## What is included

- `zerodce/model.py` — DCE-Net and iterative curve application
- `zerodce/losses.py` — the four Zero-DCE losses
- `zerodce/dataset.py` — folder-based image loader
- `zerodce/train.py` — training from scratch
- `zerodce/enhance.py` — inference on a folder of images
- `zerodce/compare_outputs.py` — compare your outputs against the official repo outputs
- `tools/make_toy_subset.py` — make a tiny dataset for quick checks

## Folder layout

```text
zerodce_from_scratch/
├── README.md
├── requirements.txt
├── zerodce/
│   ├── __init__.py
│   ├── compare_outputs.py
│   ├── dataset.py
│   ├── enhance.py
│   ├── losses.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── tools/
    └── make_toy_subset.py
```

## Dataset expectation

Put the downloaded training images inside:

```text
data/train_data/
```

The loader reads all image files recursively, so you may also use subfolders.

## Install

```bash
pip install -r requirements.txt
```

## Sequential run order

### 1) Create a tiny subset for a fast experiment

```bash
python tools/make_toy_subset.py --src data/train_data --dst data/toy_train_data --max_images 64
```

### 2) Train from scratch on the toy subset

```bash
python -m zerodce.train --data_dir data/toy_train_data --epochs 5 --batch_size 8 --save_dir runs/toy_run
```

This writes:
- `runs/toy_run/checkpoints/zerodce_epoch_###.pt`
- `runs/toy_run/train_log.csv`
- `runs/toy_run/samples/`

### 3) Enhance a few test images with your checkpoint

```bash
python -m zerodce.enhance \
  --checkpoint runs/toy_run/checkpoints/zerodce_epoch_005.pt \
  --input_dir data/test_data \
  --output_dir outputs/ours
```

### 4) Run the official repo on the same test folder

Use the official repository with the same input images and save those results to another folder, for example:

```text
outputs/official
```

### 5) Compare your outputs to the official outputs

```bash
python -m zerodce.compare_outputs --ours outputs/ours --official outputs/official --report_dir comparison_report
```

This produces:

- `comparison_report/per_image.csv`
- `comparison_report/summary.json`

The comparison is between the two sets of enhanced images, so it measures how close your implementation is to the official implementation on the same inputs.

## Notes for reproduction

The official paper and repo use the same core design:
- 7 conv layers with 32 feature maps
- no pooling and no batch normalization
- `tanh` output for the curve maps
- 8 iterations, so `24` final curve channels
- total loss:
  - `200 * L_TV`
  - `+ L_spa`
  - `+ 5 * L_col`
  - `+ 10 * L_exp` citeturn301122view3turn770689view0turn770689view1turn770689view2turn770689view3turn808003view2

That is exactly the training recipe this reimplementation follows.

## Suggested assignment write-up structure

1. State the paper and why it is zero-reference.
2. Describe the architecture and losses.
3. Mention that you reimplemented the method from scratch.
4. Train on a small subset first.
5. Show enhanced samples from your code and the official repo.
6. Report the comparison CSV and summary JSON.
7. Explain differences if your outputs are not identical.

## Citation

```bibtex
@inproceedings{Zero-DCE,
  author = {Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  title = {Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement},
  booktitle = {CVPR},
  year = {2020}
}
```
