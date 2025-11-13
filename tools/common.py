# tools/common.py
# --- path guard ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------

import os, math, json, random, hashlib, argparse, yaml
from types import SimpleNamespace
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import clip

from models.clip_gan import Generator

def set_seed(seed: int = 1337):
    import numpy as _np, random as _rand, torch as _torch
    _rand.seed(seed); _np.random.seed(seed); _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def load_cfg(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)

def load_clip_model(device):
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    model = model.float()
    for p in model.parameters():
        p.requires_grad = False
    return model

@torch.no_grad()
def encode_text(clip_model, device, texts):
    tokens = clip.tokenize(texts, truncate=True).to(device)
    feats = clip_model.encode_text(tokens).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def load_generator(gen_ckpt: str, cfg, device, ema_ckpt: str = None):
    G = Generator(z_dim=cfg.Z_DIM, cond_dim=512, image_size=cfg.IMAGE_SIZE, gf_dim=cfg.GF_DIM).to(device)
    sd = torch.load(gen_ckpt, map_location=device)
    G.load_state_dict(sd, strict=True)
    G.eval()
    if ema_ckpt and os.path.exists(ema_ckpt):
        ema_sd = torch.load(ema_ckpt, map_location=device)
        G.load_state_dict(ema_sd, strict=False)
    return G

def parse_epoch_from_ckpt(path: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(base.split("_")[-1])
    except Exception:
        return -1

def read_prompts_yaml(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "prompts" in data:
        return list(data["prompts"])
    if isinstance(data, list):
        return [str(x) for x in data]
    return []

def z_list(n: int, z_dim: int, seed: int = 123):
    rng = np.random.RandomState(seed)
    zs = rng.randn(n, z_dim).astype(np.float32)
    return torch.from_numpy(zs)

@torch.no_grad()
def generate_images(G, clip_model, device, prompts, n_per_prompt, z_tensor, trunc=1.0, image_size=64):
    texts = []
    for p in prompts:
        texts.extend([p] * n_per_prompt)
    text_feats = encode_text(clip_model, device, texts)  # [B,512]
    z = z_tensor.to(device)
    if trunc != 1.0:
        z = torch.clamp(z, -trunc, trunc)
    imgs = []
    B = z.size(0)
    bs = 64
    for i in range(0, B, bs):
        zb = z[i:i+bs]
        tfb = text_feats[i:i+bs]
        out = G(zb, tfb).detach()            # [-1,1]
        out = (out.add(1).div(2)).clamp(0,1) # [0,1]
        out = F.interpolate(out, size=image_size, mode="bilinear", align_corners=False)
        imgs.append(out.cpu())
    imgs = torch.cat(imgs, dim=0)  # [B,3,H,W]
    pil = [to_pil(imgs[i]) for i in range(imgs.size(0))]
    return pil

def to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0,1)
    t = (t * 255.0).round().byte()
    arr = t.permute(1,2,0).numpy()
    return Image.fromarray(arr)

def save_images(pils, out_dir, base_prefix="img"):
    ensure_dir(out_dir)
    paths = []
    for i, im in enumerate(pils):
        p = os.path.join(out_dir, f"{base_prefix}_{i:06d}.png")
        im.save(p)
        paths.append(p)
    return paths
