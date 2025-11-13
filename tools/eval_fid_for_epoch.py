# tools/eval_fid_for_epoch.py
# --- path guard ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------
import os, argparse, shutil
from tqdm import tqdm
import numpy as np
import torch
from torch_fidelity import calculate_metrics

from tools.common import set_seed, load_cfg, load_clip_model, load_generator, read_prompts_yaml, z_list, generate_images, save_images, ensure_dir, parse_epoch_from_ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--n_imgs", type=int, default=1020)
    ap.add_argument("--trunc", type=float, default=1.0)
    ap.add_argument("--real_dir", required=True)   # output_clip/eval/val_64
    ap.add_argument("--fake_dir", required=True)   # output_clip/eval/fake_epochXXX
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--ema_ckpt", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    clip_model = load_clip_model(device)
    G = load_generator(args.gen_ckpt, cfg, device, ema_ckpt=args.ema_ckpt)

    prompts = read_prompts_yaml(args.prompts_file)
    assert len(prompts) > 0
    n_per = int(np.ceil(args.n_imgs / len(prompts)))
    total = n_per * len(prompts)

    z = z_list(total, cfg.Z_DIM, seed=args.seed)
    pil = generate_images(G, clip_model, device, prompts, n_per, z, trunc=args.trunc, image_size=cfg.IMAGE_SIZE)
    if len(pil) > args.n_imgs:
        pil = pil[:args.n_imgs]

    if os.path.isdir(args.fake_dir):
        shutil.rmtree(args.fake_dir)
    os.makedirs(args.fake_dir, exist_ok=True)
    save_images(pil, args.fake_dir, base_prefix="fake")

    metrics = calculate_metrics(
        input1=args.fake_dir,
        input2=args.real_dir,
        cuda=torch.cuda.is_available(),
        isc=False, fid=True, kid=False, pr=False, verbose=False
    )
    fid = float(metrics["frechet_inception_distance"])
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w") as f:
        f.write(f"{fid:.6f}\n")

if __name__ == "__main__":
    main()
