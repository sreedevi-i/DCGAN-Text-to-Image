# runtime.py
import os, yaml, argparse, torch
from torchvision.utils import save_image
import clip, torch.nn.functional as F
from trainer import Trainer
from models.clip_gan import Generator

def load_cfg(path):
    with open(path, "r") as f:
        class Cfg: pass
        d = yaml.safe_load(f)
        c = Cfg()
        for k,v in d.items(): setattr(c, k, v)
        return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfg/flowers_clip.yml")
    ap.add_argument("--inference", action="store_true")
    ap.add_argument("--gen_ckpt", type=str, default="")
    ap.add_argument("--prompt", type=str, default="a red flower with yellow center")
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--trunc", type=float, default=None)
    ap.add_argument("--out", type=str, default="out.png")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    trainer = Trainer(cfg)

    if not args.inference:
        trainer.train(cfg.EPOCHS)
        return

    device = trainer.device
    netG = Generator(cfg.Z_DIM, 512, cfg.IMAGE_SIZE, cfg.GF_DIM).to(device)
    if args.gen_ckpt:
        sd = torch.load(args.gen_ckpt, map_location=device)
        netG.load_state_dict(sd, strict=True)
    else:
        netG.load_state_dict(trainer.netG.state_dict(), strict=False)
    netG.eval()

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval(); clip_model = clip_model.float()
    with torch.no_grad():
        text_feat = clip_model.encode_text(clip.tokenize([args.prompt], truncate=True).to(device)).float()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    trunc = cfg.Z_TRUNC_INFER if args.trunc is None else args.trunc
    torch.manual_seed(cfg.SEED)
    z = torch.randn(args.n_samples, cfg.Z_DIM, device=device) * trunc
    with torch.no_grad():
        imgs = netG(z, text_feat.repeat(args.n_samples, 1))
    save_image((imgs+1)/2, args.out, nrow=int(args.n_samples**0.5) or 1)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
