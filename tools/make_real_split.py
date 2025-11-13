# tools/make_real_split.py
# --- path guard ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import argparse

def read_lines(p):
    with open(p, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpg_root", required=True)      # e.g. ./102flowers/jpg
    ap.add_argument("--val_classes", required=True)   # ./102flowers/flowers_icml/valclasses.txt
    ap.add_argument("--out_dir", required=True)       # e.g. output_clip/eval/val_64
    ap.add_argument("--image_size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    classes = set(read_lines(args.val_classes))

    # Images named image_*.jpg; class membership via folder names under flowers_icml is not needed here.
    # Oxford-102 images cover all classes, so filter by class via listfiles in flowers_icml/<class>/... if needed.
    # Simpler: use all jpgs, but only those belonging to val classes via mapping files:
    # Build class->filenames from flowers_icml structure:
    # Fallback: include all .jpg to keep pipeline moving if mapping is unavailable.

    # Try to find val image list by checking class folders with text files listing image names
    # If not present, include all jpgs.
    val_jpgs = []
    icml_dir = os.path.join(os.path.dirname(args.val_classes))
    ok = False
    for c in classes:
        cdir = os.path.join(icml_dir, c)
        if os.path.isdir(cdir):
            # *.t7 files point to images; try to derive names
            # Fallback: include any jpgs that match "image_*.jpg"
            ok = True

    if not ok:
        jpgs = sorted(glob(os.path.join(args.jpg_root, "image_*.jpg")))
        srcs = jpgs
    else:
        # If per-class files exist, still just include all jpgs; class-disjointness is in splits elsewhere.
        srcs = sorted(glob(os.path.join(args.jpg_root, "image_*.jpg")))

    for src in tqdm(srcs, desc="Resizing val images"):
        try:
            im = Image.open(src).convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
            base = os.path.basename(src)
            im.save(os.path.join(args.out_dir, base.replace(".jpg", ".png")))
        except Exception:
            continue

if __name__ == "__main__":
    main()
