# tools/eval_clip_over_epoch_safe.py
import os, sys, glob, csv, argparse, time
import yaml
import torch
import torch.nn.functional as F
import clip

# Project-relative imports (works when run from project root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_gan import Generator  # uses your existing Generator

def load_cfg(path):
    import yaml
    with open(path, "r") as f:
        return argparse.Namespace(**yaml.safe_load(f))

def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

@torch.no_grad()
def clip_score(model, preprocess, device, images, text_tokens):
    # images in [-1,1] from G -> [0,1] -> resize 224
    imgs_01 = (images.clamp(-1, 1) + 1) / 2
    imgs_224 = F.interpolate(imgs_01, size=224, mode='bilinear', align_corners=False)
    img_feats = model.encode_image(imgs_224).float()
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    txt_feats = model.encode_text(text_tokens).float()
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    # mean over prompts if multiple; here prompts are batched
    sims = (img_feats @ txt_feats.T)  # [B, P]
    # average sim per image across prompts, then mean over batch
    return sims.mean().item()

def load_generator(gen_ckpt, cfg, device):
    netG = Generator(
        z_dim=cfg.Z_DIM,
        cond_dim=512,          # CLIP ViT-B/32
        image_size=cfg.IMAGE_SIZE,
        gf_dim=cfg.GF_DIM
    ).to(device)
    sd = torch.load(gen_ckpt, map_location=device)
    # Allow strict=False in case of missing buffers
    netG.load_state_dict(sd, strict=False)
    netG.eval()
    return netG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--samples_per_prompt", type=int, default=4)
    ap.add_argument("--trunc", type=float, default=1.0)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--samples_csv", required=True)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--max_ckpts", type=int, default=0, help="limit number of checkpoints (0 = all)")
    args = ap.parse_args()

    print("Starting eval_clip_over_epoch_safe...", flush=True)
    print(f"ckpt_dir = {args.ckpt_dir}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    cfg = load_cfg(args.cfg)

    # Prompts
    with open(args.prompts_file, "r") as f:
        y = yaml.safe_load(f)
    prompts = y.get("prompts", [])
    if not prompts:
        print("No prompts found in YAML. Exiting.", flush=True)
        return

    # Checkpoints
    pattern = "genEMA_*.pth" if args.ema else "gen_*.pth"
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, pattern)))
    if not ckpts:
        print(f"No checkpoints matching {pattern}. Exiting.", flush=True)
        return
    if args.max_ckpts:
        ckpts = ckpts[:args.max_ckpts]

    # CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    clip_model = clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Tokenize prompts once
    text_tokens = clip.tokenize(prompts, truncate=True).to(device)

    # Prepare logs
    ensure_dir(args.summary_csv)
    ensure_dir(args.samples_csv)
    summary_f = open(args.summary_csv, "w", newline="")
    samples_f = open(args.samples_csv, "w", newline="")
    sum_w = csv.writer(summary_f); sam_w = csv.writer(samples_f)
    sum_w.writerow(["epoch","ckpt_name","mean_clip_score","num_images","num_prompts","samples_per_prompt","trunc"])
    sam_w.writerow(["epoch","ckpt_name","prompt","sample_idx","clip_score"])

    # Evaluate
    z_dim = cfg.Z_DIM
    total = len(ckpts)
    for i, ck in enumerate(ckpts, 1):
        t0 = time.time()
        # Infer epoch number
        base = os.path.basename(ck)
        ep = None
        try:
            ep = int(base.split("_")[-1].split(".")[0])
        except:
            ep = -1

        print(f"[{i}/{total}] Loading {base}", flush=True)
        netG = load_generator(ck, cfg, device)

        # Generate per-prompt samples, score, and record
        all_scores = []
        for p_i, _ in enumerate(prompts):
            # One batch per prompt: generate N images with the SAME prompt embedding
            with torch.no_grad():
                z = torch.randn(args.samples_per_prompt, z_dim, device=device) * args.trunc
                txt_feats = clip_model.encode_text(text_tokens[p_i:p_i+1]).float()
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                txt_feats = txt_feats.repeat(args.samples_per_prompt,1)  # [N,512]
                imgs = netG(z, txt_feats)
                # Score each image against its own prompt
                imgs_01 = (imgs.clamp(-1,1)+1)/2
                imgs_224 = F.interpolate(imgs_01, size=224, mode='bilinear', align_corners=False)
                img_feats = clip_model.encode_image(imgs_224).float()
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                sc = (img_feats * (txt_feats / txt_feats.norm(dim=-1, keepdim=True))).sum(dim=-1)  # [N]
                for s_idx, s in enumerate(sc.tolist()):
                    sam_w.writerow([ep, base, prompts[p_i], s_idx, s])
                all_scores.extend(sc.tolist())

        mean_score = float(sum(all_scores)/max(1,len(all_scores)))
        sum_w.writerow([ep, base, f"{mean_score:.6f}", len(all_scores), len(prompts), args.samples_per_prompt, args.trunc])
        summary_f.flush(); samples_f.flush()
        print(f"  mean CLIP score = {mean_score:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    summary_f.close(); samples_f.close()
    print("Done. Wrote:", args.summary_csv, "and", args.samples_csv, flush=True)

if __name__ == "__main__":
    main()
