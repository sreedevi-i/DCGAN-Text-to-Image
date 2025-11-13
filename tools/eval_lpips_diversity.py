
import os, sys, argparse, itertools, csv, warnings
import numpy as np
import torch
import torchvision.transforms as T
import lpips
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.common import (
    set_seed, load_cfg, load_clip_model, load_generator,
    read_prompts_yaml, z_list, generate_images, ensure_dir
)

warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")

def pil_to_tensor_m11(pils, device):
    to_tensor = T.ToTensor()
    xs = []
    for im in pils:
        x = to_tensor(im)
        if x.size(0) == 1:
            x = x.repeat(3,1,1)
        if x.size(0) == 4:
            x = x[:3]
        x = x*2.0 - 1.0
        xs.append(x)
    return torch.stack(xs, 0).to(device)

def mean_pairwise_lpips(x, lpips_fn, batch=16):
    idxs = list(itertools.combinations(range(x.size(0)), 2))
    if not idxs:
        return float("nan")
    vals = []
    for i in range(0, len(idxs), batch):
        chunk = idxs[i:i+batch]
        a = torch.stack([x[a_] for a_,_ in chunk], 0)
        b = torch.stack([x[b_] for _,b_ in chunk], 0)
        with torch.no_grad():
            d = lpips_fn(a, b).flatten().detach().cpu()
        vals.append(d)
    return float(torch.cat(vals, 0).mean().item())

def iter_ckpts(ckpt_dir):
    fs = [f for f in os.listdir(ckpt_dir) if f.startswith("gen_") and f.endswith(".pth")]
    def ep(f):
        try: return int(os.path.splitext(f)[0].split("_")[-1])
        except: return -1
    fs = sorted(fs, key=ep)
    for f in fs:
        e = ep(f)
        g = os.path.join(ckpt_dir, f)
        ema = os.path.join(ckpt_dir, f"genEMA_{e:03d}.pth")
        yield e, g, (ema if os.path.exists(ema) else None)

def run_for_checkpoint(gen_ckpt, ema_ckpt, cfg, device, prompts, n_per, trunc, seed, lpips_fn):
    clip_model = load_clip_model(device)
    G = load_generator(gen_ckpt, cfg, device, ema_ckpt=ema_ckpt)
    total = len(prompts)*n_per
    Z = z_list(total, cfg.Z_DIM, seed=seed)
    pils = generate_images(G, clip_model, device, prompts, n_per, Z, trunc=trunc, image_size=cfg.IMAGE_SIZE)
    x = pil_to_tensor_m11(pils, device)
    per_prompt = []
    i = 0
    for _ in prompts:
        chunk = x[i:i+n_per]
        i += n_per
        if chunk.size(0) >= 2:
            per_prompt.append(mean_pairwise_lpips(chunk, lpips_fn, batch=16))
    if not per_prompt:
        return float("nan")
    return float(np.mean(per_prompt))

def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt_dir")
    grp.add_argument("--gen_ckpt")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_ckpt")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--samples_per_prompt", type=int, default=8)
    ap.add_argument("--trunc", type=float, default=1.0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    prompts = read_prompts_yaml(args.prompts_file)
    if len(prompts) == 0:
        print(f"[ERR] No prompts in {args.prompts_file}")
        return

    ensure_dir(os.path.dirname(args.out_csv))
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    rows = []
    if args.gen_ckpt:
        score = run_for_checkpoint(
            gen_ckpt=args.gen_ckpt,
            ema_ckpt=args.ema_ckpt,
            cfg=cfg, device=device, prompts=prompts,
            n_per=args.samples_per_prompt, trunc=args.trunc, seed=args.seed,
            lpips_fn=lpips_fn
        )
        ep = Path(args.gen_ckpt).stem.split("_")[-1]
        rows.append((int(ep) if ep.isdigit() else -1, score, 1 if args.ema_ckpt else 0,
                     args.trunc, args.samples_per_prompt, len(prompts)))
        print(f"[LPIPS] epoch {ep} | mean within-prompt = {score:.4f} | EMA={'yes' if args.ema_ckpt else 'no'}")
    else:
        ckpts = list(iter_ckpts(args.ckpt_dir))
        print(f"[INFO] Found {len(ckpts)} checkpoints in {args.ckpt_dir}")
        for e, g, ema in tqdm(ckpts, desc="Epochs"):
            use_ema_ckpt = (ema if args.ema else None)
            score = run_for_checkpoint(
                gen_ckpt=g, ema_ckpt=use_ema_ckpt,
                cfg=cfg, device=device, prompts=prompts,
                n_per=args.samples_per_prompt, trunc=args.trunc, seed=args.seed,
                lpips_fn=lpips_fn
            )
            rows.append((e, score, 1 if use_ema_ckpt else 0,
                         args.trunc, args.samples_per_prompt, len(prompts)))
            print(f"[LPIPS] epoch {e:03d} | mean within-prompt = {score:.4f} | EMA={'yes' if use_ema_ckpt else 'no'}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lpips_within_mean", "ema_used", "trunc", "samples_per_prompt", "n_prompts"])
        w.writerows(rows)
    print(f"[OK] Wrote {args.out_csv} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
