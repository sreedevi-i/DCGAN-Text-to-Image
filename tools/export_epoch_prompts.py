import os, sys, argparse, yaml, math, random
from datetime import datetime
import torch, torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from PIL import Image
import clip
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_gan import Generator

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def read_prompts_yaml(p):
    with open(p, "r") as f:
        y = yaml.safe_load(f)
    return [str(x) for x in y.get("prompts", [])]

def slugify(t):
    s = "".join(c.lower() if c.isalnum() else "_" for c in t).strip("_")
    return "_".join(s.split("_")[:10])[:80] or "prompt"

@torch.no_grad()
def encode_texts(model, device, texts):
    toks = clip.tokenize(texts, truncate=True).to(device)
    feats = model.encode_text(toks).float()
    return feats / feats.norm(dim=-1, keepdim=True)

@torch.no_grad()
def clip_scores(model, device, imgs, text_feats):
    imgs_01 = (imgs + 1) / 2
    imgs_224 = F.interpolate(imgs_01, size=224, mode="bilinear", align_corners=False)
    imgf = model.encode_image(imgs_224).float()
    imgf = imgf / imgf.norm(dim=-1, keepdim=True)
    return (imgf * text_feats).sum(dim=-1)

def load_generator(gen_ckpt, image_size, z_dim, gf_dim, device, ema_ckpt=None):
    G = Generator(z_dim=z_dim, cond_dim=512, image_size=image_size, gf_dim=gf_dim).to(device)
    sd = torch.load(gen_ckpt, map_location=device)
    G.load_state_dict(sd, strict=True)
    if ema_ckpt and os.path.exists(ema_ckpt):
        ema_sd = torch.load(ema_ckpt, map_location=device)
        G.load_state_dict(ema_sd, strict=False)
    G.eval()
    return G

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--epochs", required=True)  # e.g. "175,120,176,122"
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_prompt", type=int, default=64)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--trunc", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ema", action="store_true")
    args = ap.parse_args()

    # Minimal cfg loader
    import yaml as _y
    with open(args.cfg, "r") as f:
        cfg = _y.safe_load(f)
    image_size = int(cfg.get("IMAGE_SIZE", 64))
    z_dim      = int(cfg.get("Z_DIM", 100))
    gf_dim     = int(cfg.get("GF_DIM", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = read_prompts_yaml(args.prompts_file)
    assert len(prompts) > 0

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    clip_model = clip_model.float()

    epochs = [int(e.strip()) for e in args.epochs.split(",") if e.strip()]
    for e in epochs:
        gen_p = os.path.join(args.ckpt_dir, f"gen_{e:03d}.pth")
        ema_p = os.path.join(args.ckpt_dir, f"genEMA_{e:03d}.pth") if args.ema else None
        if not os.path.exists(gen_p):
            print(f"[skip] {gen_p} not found")
            continue

        G = load_generator(gen_p, image_size, z_dim, gf_dim, device, ema_ckpt=ema_p if (ema_p and os.path.exists(ema_p)) else None)

        epoch_dir = os.path.join(args.out_dir, f"epoch{e:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        best_images = []
        for ptxt in prompts:
            tf = encode_texts(clip_model, device, [ptxt])  # [1,512]
            z = torch.randn(args.n_per_prompt, z_dim, device=device) * args.trunc
            tf_rep = tf.repeat(args.n_per_prompt, 1)

            imgs = G(z, tf_rep).clamp(-1, 1)  # [N,3,H,W]
            sc = clip_scores(clip_model, device, imgs, tf_rep)  # [N]

            topk = min(args.topk, imgs.size(0))
            vals, idx = torch.topk(sc, k=topk, largest=True)
            chosen = imgs[idx]  # [topk,3,H,W]

            # Save per-prompt best image (first of top-k)
            prompt_png = os.path.join(epoch_dir, f"{slugify(ptxt)}.png")
            save_image((chosen[0] + 1)/2, prompt_png)
            best_images.append(chosen[0])

        # Combined grid for the epoch
        grid = make_grid(torch.stack(best_images, dim=0), nrow=math.ceil(len(prompts)/2))
        grid_png = os.path.join(args.out_dir, f"epoch{e:03d}_grid.png")
        save_image((grid + 1)/2, grid_png)
        print(f"[done] epoch {e:03d}: saved {len(prompts)} prompt images and {grid_png}")

if __name__ == "__main__":
    main()
