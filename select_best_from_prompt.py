# select_best_from_prompt.py
import os, argparse, yaml, math, torch
import clip, torch.nn.functional as F
from torchvision.utils import save_image
from models.clip_gan import Generator

def load_cfg(path):
    with open(path, "r") as f:
        class Cfg: pass
        d = yaml.safe_load(f)
        c = Cfg()
        for k,v in d.items(): setattr(c, k, v)
        return c

@torch.no_grad()
def score_with_clip(clip_model, imgs_01, text_feat):
    imgs_224 = F.interpolate(imgs_01, size=224, mode='bilinear', align_corners=False)
    img_feats = clip_model.encode_image(imgs_224).float()
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    sims = (img_feats * text_feat).sum(dim=-1)
    return sims

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfg/flowers_clip.yml")
    ap.add_argument("--gen_ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--trunc", type=float, default=0.7)
    ap.add_argument("--out", default="best.png")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator(cfg.Z_DIM, 512, cfg.IMAGE_SIZE, cfg.GF_DIM).to(device)
    netG.load_state_dict(torch.load(args.gen_ckpt, map_location=device), strict=True)
    netG.eval()

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval(); clip_model = clip_model.float()
    with torch.no_grad():
        text_feat = clip_model.encode_text(clip.tokenize([args.prompt], truncate=True).to(device)).float()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    torch.manual_seed(cfg.SEED)
    z = torch.randn(args.n, cfg.Z_DIM, device=device) * args.trunc
    with torch.no_grad():
        imgs = netG(z, text_feat.repeat(args.n, 1))
        imgs_01 = (imgs + 1) / 2
        sims = score_with_clip(clip_model, imgs_01, text_feat.repeat(imgs_01.size(0), 1))
    topk = min(args.topk, args.n)
    idx = torch.topk(sims, k=topk).indices
    grid = imgs_01[idx]
    save_image(grid, args.out, nrow=int(math.sqrt(topk)) or 1)
    print(f"Saved top-{topk} grid to {args.out}")

if __name__ == "__main__":
    main()
