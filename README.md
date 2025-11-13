# DCGAN-Text-to-Image - CLIP-Guided DCGAN on Oxford-102 Flowers

Prompt-controlled text-to-image synthesis on Oxford-102 Flowers using a DCGAN trained with CLIP guidance. Training, sampling, ranking, and evaluation utilities are provided, with a Colab-friendly workflow and a Jupyter notebook (`CLIP_GUIDED_DCGAN.ipynb`) that ties everything together.

---

## Repository Structure 

```
DCGAN-Text-to-Image/
├─ cfg/
│  └─ flowers_clip.yml          # All hyperparameters and paths
├─ models/
│  └─ clip_gan.py               # Generator & Discriminator (CLIP-conditioned)
├─ tools/                       # Data prep & evaluation utilities
│  ├─ convert_flowers_to_hd5_script.py  # Build flowers.hdf5 from raw data
│  ├─ make_real_split.py        # Resize real val images for FID
│  ├─ eval_clip_over_epoch.py   # CLIP alignment vs. epoch (CSV summaries)
│  ├─ eval_fid_for_epoch.py     # FID for a specific epoch
│  ├─ eval_lpips_diversity.py   # LPIPS diversity (within-prompt)
│  └─ __init__.py               # Package marker for `python -m tools.*`
├─ txt2image_dataset.py         # HDF5-backed dataset loader
├─ trainer.py                   # Training loop (saves CSV logs & checkpoints)
├─ runtime.py                   # Small runner that loads cfg and calls Trainer
├─ select_best_gen.py           # Rank gen_*.pth by CLIP alignment
├─ select_best_from_prompt.py   # Best-of-N sampling for a prompt
├─ class_names.py               # Optional mapping for class prompts
└─ CLIP_GUIDED_DCGAN.ipynb      # End-to-end notebook (training + eval + plots)
```

---

## Data Preparation

Download
---
Oxford 102 flowers image dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

Run convert_flowers_to_hd5_script.py to obtain the flowers.hdf5 file
----

Required assets (place them under `./102flowers/` and alongside):

* Oxford-102 Flowers images at `./102flowers/jpg/`
* **flowers_icml** split lists:

  * `./102flowers/trainclasses.txt`
  * `./102flowers/valclasses.txt`
  * `./102flowers/testclasses.txt`
* **text_c10** captions directory (10 human captions per image), e.g. `./text_c10/`

**Create `flowers.hdf5`:**

```bash
python tools/convert_flowers_to_hd5_script.py \
  --jpg_root 102flowers/jpg \
  --text_root text_c10 \
  --train_classes 102flowers/trainclasses.txt \
  --val_classes 102flowers/valclasses.txt \
  --test_classes 102flowers/testclasses.txt \
  --out flowers.hdf5 \
  --caps_per_image 3
```

*(For FID evaluation)* Create a resized real validation set at 64×64:

```bash
python -m tools.make_real_split \
  --jpg_root 102flowers/jpg \
  --val_classes 102flowers/valclasses.txt \
  --out_dir output_clip/eval/val_64 \
  --image_size 64
```

---

## Environment Setup

```bash
pip install torch torchvision
pip install ftfy regex tqdm h5py pyyaml pillow matplotlib numpy pandas
pip install git+https://github.com/openai/CLIP.git
pip install torch-fidelity lpips
```


---

## Configuration

 `cfg/flowers_clip.yml` is used to store paths and hyperparameters:

```yaml
FLOWERS_HDF5: "flowers.hdf5"

BATCH_SIZE: 64
IMAGE_SIZE: 64
WORKERS: 2

Z_DIM: 100
GF_DIM: 64
DF_DIM: 64

LR_G: 0.0002
LR_D: 0.0001

LAMBDA_CLIP: 0.2

OUT_DIR: "output_clip"
EPOCHS: 200


```

---

## Run Training

**Python:**

```bash
python runtime.py --cfg cfg/flowers_clip.yml
```

**Colab (Drive):**

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/DCGAN-Text-to-Image
!python runtime.py --cfg cfg/flowers_clip.yml
```

Artifacts will be saved under `OUT_DIR`:

* `output_clip/checkpoints/` → `gen_XXX.pth`, `disc_XXX.pth`, `state_XXX.pth`, `genEMA_XXX.pth`
* `output_clip/logs/` → `train_YYYYMMDD_HHMMSS.csv` plus evaluation CSVs
* `output_clip/figs/` → plots created 

The trainer auto-resumes from `state_*.pth` .

---

## Sampling & Ranking 

**Best-of-N for a single prompt:**

```bash
python select_best_from_prompt.py \
  --cfg cfg/flowers_clip.yml \
  --gen_ckpt output_clip/checkpoints/gen_195.pth \
  --ema_ckpt output_clip/checkpoints/genEMA_195.pth \
  --prompt "a bright yellow sunflower with a dark center" \
  --n 256 --topk 1 --trunc 0.7 \
  --out exports/sunflower_best1_e195.png
```

**Rank checkpoints by CLIP alignment:**

```bash
python select_best_gen.py \
  --cfg cfg/flowers_clip.yml \
  --ckpt_dir output_clip/checkpoints \
  --prompt "a red flower with a yellow center" \
  --n 64 --trunc 0.8
```

---

## Evaluation 

**CLIP alignment vs. epoch** (requires a YAML prompts list, e.g. `output_clip/eval/prompts.yaml`):

```bash
PYTHONPATH=. python -m tools.eval_clip_over_epoch \
  --ckpt_dir output_clip/checkpoints \
  --cfg cfg/flowers_clip.yml \
  --prompts_file output_clip/eval/prompts.yaml \
  --samples_per_prompt 8 \
  --trunc 0.9 \
  --summary_csv output_clip/logs/eval_clip_summary.csv \
  --samples_csv output_clip/logs/eval_clip_samples_all.csv \
  --ema
```

**FID for a chosen epoch** (needs `output_clip/eval/val_64`):

```bash
E=195
PYTHONPATH=. python -m tools.eval_fid_for_epoch \
  --gen_ckpt output_clip/checkpoints/gen_${E}.pth \
  --cfg cfg/flowers_clip.yml \
  --prompts_file output_clip/eval/prompts.yaml \
  --n_imgs 1020 \
  --trunc 0.9 \
  --real_dir output_clip/eval/val_64 \
  --fake_dir output_clip/eval/fake_epoch${E} \
  --out_txt output_clip/logs/fid/fid_epoch${E}.txt \
  --ema_ckpt output_clip/checkpoints/genEMA_${E}.pth
```

**LPIPS diversity command** :

```bash
PYTHONPATH=. python -m tools.eval_lpips_diversity \
  --ckpt_dir output_clip/checkpoints \
  --cfg cfg/flowers_clip.yml \
  --prompts_file output_clip/eval/prompts.yaml \
  --samples_per_prompt 8 \
  --trunc 0.9 \
  --out_csv output_clip/logs/eval_lpips_diversity.csv \
  --ema_ckpt auto
```

---

## Notes

* Used `state_*.pth` for resuming and `genEMA_*.pth` for clean sampling.
* The notebook `CLIP_GUIDED_DCGAN.ipynb` contains end-to-end cells for training, evaluation, and plotting with the same relative file layout.
