import os
import re
import torch
import torch.nn.functional as F
import yaml

from types import SimpleNamespace

import clip
from models.clip_gan import Generator


def load_cfg(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)


def resolve_path(base_dir, p):
    if os.path.isabs(p):
        return p
    return os.path.join(base_dir, p)


@torch.no_grad()
def score_checkpoint(gen_ckpt_path, cfg, device, clip_model, test_prompts, n_samples=8):
    """
    Compute mean CLIP(image, text) similarity for one generator checkpoint.
    Returns float score. Raises on load/forward errors.
    """
    # Load generator weights
    netG = Generator(
        z_dim=cfg.Z_DIM,
        cond_dim=512,
        image_size=cfg.IMAGE_SIZE,
        gf_dim=cfg.GF_DIM
    ).to(device)

    state = torch.load(gen_ckpt_path, map_location=device)
    netG.load_state_dict(state)
    netG.eval()

    sims = []

    for prompt in test_prompts:
        # Text features
        tokens = clip.tokenize([prompt], truncate=True).to(device)
        txt_feat = clip_model.encode_text(tokens)
        txt_feat = txt_feat.float()
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)  # [1,512]

        # Generate images
        z = torch.randn(n_samples, cfg.Z_DIM, device=device)
        cond = txt_feat.repeat(n_samples, 1)
        fake = netG(z, cond)                  # [-1,1]
        fake_01 = (fake + 1) / 2              # [0,1]
        fake_224 = F.interpolate(
            fake_01, size=224,
            mode="bilinear", align_corners=False
        )

        # Image features
        img_feat = clip_model.encode_image(fake_224)
        img_feat = img_feat.float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # Cosine similarity with text
        sim = (img_feat * txt_feat).sum(dim=-1)  # [N]
        sims.append(sim.mean().item())

    return sum(sims) / len(sims)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load config
    cfg_path = os.path.join(base_dir, "cfg", "flowers_clip.yml")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_cfg(cfg_path)

    # Resolve checkpoint dir
    out_dir = resolve_path(base_dir, cfg.OUT_DIR)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"Checkpoint dir not found: {ckpt_dir}")

    # Collect available gen_XXX.pth
    gen_ckpts = []
    for f in os.listdir(ckpt_dir):
        m = re.match(r"gen_(\d+)\.pth$", f)
        if m:
            epoch = int(m.group(1))
            path = os.path.join(ckpt_dir, f)
            if os.path.isfile(path):
                gen_ckpts.append((epoch, path))

    if not gen_ckpts:
        raise SystemExit("No gen_XXX.pth checkpoints found.")

    gen_ckpts.sort(key=lambda x: x[0])
    print(f"Found {len(gen_ckpts)} generator checkpoints.")

    # Device & CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    clip_model = clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Test prompts: mix of generic + class-like
    test_prompts = [
        "a bright red flower with a yellow center",
        "a yellow flower with a dark center",
        "a pink flower with many thin petals",
        "a white daisy flower with a yellow center",
        "a yellow sunflower with a dark center",
        "a red rose flower with many petals",
    ]

    results = []
    print("Scoring checkpoints...")

    for epoch, path in gen_ckpts:
        try:
            score = score_checkpoint(
                gen_ckpt_path=path,
                cfg=cfg,
                device=device,
                clip_model=clip_model,
                test_prompts=test_prompts,
                n_samples=8,
            )
            results.append((epoch, score))
            print(f"Epoch {epoch:3d}: CLIP score = {score:.4f}")
        except (OSError, RuntimeError, torch.serialization.UnpicklingError) as e:
            # Handle broken Drive mount or corrupted checkpoint gracefully
            print(f"Skipping gen_{epoch:03d}.pth due to error: {e}")
            continue

    if not results:
        raise SystemExit("No valid checkpoints could be scored (all failed to load).")

    # Sort by score: higher is better
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Top checkpoints by CLIP alignment ===")
    top_k = min(10, len(results))
    for i in range(top_k):
        epoch, score = results[i]
        print(f"Epoch {epoch:3d}: {score:.4f}  -> gen_{epoch:03d}.pth")

    best_epoch, best_score = results[0]
    print(f"\nBest checkpoint: gen_{best_epoch:03d}.pth (score {best_score:.4f})")


if __name__ == "__main__":
    main()
