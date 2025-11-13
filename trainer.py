# trainer.py
import os
import csv
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from txt2image_dataset import Text2ImageDataset
from models.clip_gan import Generator, Discriminator

try:
    from class_names import CLASS_NAME_MAP
except Exception:
    CLASS_NAME_MAP = {}

# ----------------------- utils -----------------------

def weights_init(m):
    name = m.__class__.__name__
    if "Conv" in name:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}

    @torch.no_grad()
    def reset_from(self, model: nn.Module):
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if torch.is_floating_point(v)}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if not torch.is_floating_point(v):
                continue
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)

def diffaugment(x):
    B, C, H, W = x.size()
    theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], device=x.device).float().repeat(B, 1, 1)
    tx = torch.randint(-2, 3, (B,), device=x.device).float() / (W / 2)
    ty = torch.randint(-2, 3, (B,), device=x.device).float() / (H / 2)
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    if torch.rand(1).item() < 0.5:
        cut = max(1, int(0.12 * H))
        cx = torch.randint(cut, W - cut, (B,), device=x.device)
        cy = torch.randint(cut, H - cut, (B,), device=x.device)
        for i in range(B):
            x[i, :, cy[i] - cut:cy[i] + cut, cx[i] - cut:cx[i] + cut] = 0
    return x

def clip_warmup_weight(epoch: int, base: float, warm: int = 10):
    return base * min(1.0, epoch / max(1, warm))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------------------- Trainer ----------------------------

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # seed
        self.seed = getattr(cfg, "SEED", 1337)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        self.cpu_rng = torch.Generator(device="cpu").manual_seed(self.seed)

        # paths
        self.hdf5_path = cfg.FLOWERS_HDF5 if os.path.isabs(cfg.FLOWERS_HDF5) \
            else os.path.join(self.base_dir, cfg.FLOWERS_HDF5)
        self.out_dir = cfg.OUT_DIR if os.path.isabs(cfg.OUT_DIR) \
            else os.path.join(self.base_dir, cfg.OUT_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logs_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        # csv log
        run_tag = time.strftime("%Y%m%d_%H%M%S")
        self.train_log_csv = os.path.join(self.logs_dir, f"train_{run_tag}.csv")
        with open(self.train_log_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["epoch", "d_loss", "g_adv", "clip_loss", "g_total", "steps"])

        # hyperparams
        self.image_size = cfg.IMAGE_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.workers    = cfg.WORKERS
        self.z_dim      = cfg.Z_DIM
        self.gf_dim     = cfg.GF_DIM
        self.df_dim     = cfg.DF_DIM
        self.lr_g       = cfg.LR_G
        self.lr_d       = cfg.LR_D
        self.lambda_clip = cfg.LAMBDA_CLIP
        self.class_mix_prob = getattr(cfg, "CLASS_MIX_PROB", 0.5)
        self.ema_decay  = getattr(cfg, "EMA_DECAY", 0.999)
        self.clip_warm  = getattr(cfg, "CLIP_WARMUP", 10)
        self.use_diffaug = getattr(cfg, "DIFFAUG", True)
        self.save_every = getattr(cfg, "SAVE_EVERY", 1)
        self.z_trunc    = getattr(cfg, "Z_TRUNC_TRAIN", 1.0)

        # data
        self.train_dataset = Text2ImageDataset(
            hdf5_path=self.hdf5_path,
            split="train",
            image_size=self.image_size
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
            generator=self.cpu_rng,
            worker_init_fn=seed_worker,
            persistent_workers=False
        )

        # CLIP encoders (frozen, float32)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        self.clip_model = self.clip_model.float()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.cond_dim = 512

        # models
        self.netG = Generator(
            z_dim=self.z_dim,
            cond_dim=self.cond_dim,
            image_size=self.image_size,
            gf_dim=self.gf_dim
        ).to(self.device)
        self.netD = Discriminator(
            cond_dim=self.cond_dim,
            image_size=self.image_size,
            df_dim=self.df_dim
        ).to(self.device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # EMA
        self.ema = EMA(self.netG, decay=self.ema_decay)
        self.ema.reset_from(self.netG)

        # opt & loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.optG = optim.Adam(self.netG.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.netD.parameters(), lr=self.lr_d, betas=(0.5, 0.999))

        self.real_label_value = 0.9
        self.fake_label_value = 0.0

        # resume
        self.start_epoch = 1
        self._maybe_resume()
        self._sync_lrs_with_cfg()

    # --------------- encoders / losses ---------------

    @torch.no_grad()
    def encode_text(self, texts):
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        feats = self.clip_model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def clip_image_text_loss(self, fake_imgs, text_features):
        imgs_01 = (fake_imgs + 1) / 2
        imgs_224 = F.interpolate(imgs_01, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(imgs_224).float()
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        sim = (img_features * text_features).sum(dim=-1)
        return 1.0 - sim.mean()

    # --------------- checkpointing ---------------

    def _maybe_resume(self):
        state_files = [f for f in os.listdir(self.ckpt_dir)
                       if f.startswith("state_") and f.endswith(".pth")]
        if state_files:
            def ep(x):
                try:
                    return int(os.path.splitext(x)[0].split("_")[-1])
                except Exception:
                    return -1
            last = max(state_files, key=ep)
            path = os.path.join(self.ckpt_dir, last)
            state = torch.load(path, map_location=self.device)

            self.netG.load_state_dict(state["netG"])
            self.netD.load_state_dict(state["netD"])
            self.optG.load_state_dict(state["optG"])
            self.optD.load_state_dict(state["optD"])
            self.start_epoch = int(state["epoch"]) + 1

            ema_path = os.path.join(self.ckpt_dir, f"genEMA_{int(state['epoch']):03d}.pth")
            if os.path.exists(ema_path):
                sd = torch.load(ema_path, map_location="cpu")
                self.ema.shadow = {k: v.to(self.device) for k, v in sd.items()
                                   if torch.is_floating_point(v)}
            else:
                self.ema.reset_from(self.netG)

            print(f"Resumed full state from {last}, starting at epoch {self.start_epoch}")
            return

        gen_files = [f for f in os.listdir(self.ckpt_dir)
                     if f.startswith("gen_") and f.endswith(".pth")]
        if gen_files:
            def ep(x):
                try:
                    return int(os.path.splitext(x)[0].split("_")[-1])
                except Exception:
                    return -1
            last_gen = max(gen_files, key=ep)
            e = ep(last_gen)
            g_path = os.path.join(self.ckpt_dir, f"gen_{e:03d}.pth")
            d_path = os.path.join(self.ckpt_dir, f"disc_{e:03d}.pth")
            if os.path.exists(g_path):
                self.netG.load_state_dict(torch.load(g_path, map_location=self.device))
            if os.path.exists(d_path):
                self.netD.load_state_dict(torch.load(d_path, map_location=self.device))
            ema_path = os.path.join(self.ckpt_dir, f"genEMA_{e:03d}.pth")
            if os.path.exists(ema_path):
                sd = torch.load(ema_path, map_location="cpu")
                self.ema.shadow = {k: v.to(self.device) for k, v in sd.items()
                                   if torch.is_floating_point(v)}
            else:
                self.ema.reset_from(self.netG)
            self.start_epoch = e + 1
            print(f"Resumed weights from gen_{e:03d}.pth / disc_{e:03d}.pth, starting at epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch: int):
        g_path = os.path.join(self.ckpt_dir, f"gen_{epoch:03d}.pth")
        d_path = os.path.join(self.ckpt_dir, f"disc_{epoch:03d}.pth")
        torch.save(self.netG.state_dict(), g_path)
        torch.save(self.netD.state_dict(), d_path)

        state = {
            "epoch": epoch,
            "netG": self.netG.state_dict(),
            "netD": self.netD.state_dict(),
            "optG": self.optG.state_dict(),
            "optD": self.optD.state_dict(),
        }
        s_path = os.path.join(self.ckpt_dir, f"state_{epoch:03d}.pth")
        torch.save(state, s_path)

        ema_path = os.path.join(self.ckpt_dir, f"genEMA_{epoch:03d}.pth")
        torch.save({k: v.detach().cpu() for k, v in self.ema.shadow.items()}, ema_path)

    def _sync_lrs_with_cfg(self):
        for pg in self.optG.param_groups:
            pg["lr"] = self.lr_g
        for pg in self.optD.param_groups:
            pg["lr"] = self.lr_d

    # --------------- training ---------------

    def train(self, epochs: int):
        if self.start_epoch > epochs:
            print(f"Already trained up to epoch {self.start_epoch - 1}. Nothing to do.")
            return

        for epoch in range(self.start_epoch, epochs + 1):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
            sum_d = sum_gadv = sum_clip = sum_gtot = 0.0
            steps = 0

            lam_clip = clip_warmup_weight(epoch, self.lambda_clip, warm=self.clip_warm)

            for batch in pbar:
                imgs = batch["image"].to(self.device).float()
                classes = batch["class"]
                caps = batch["caption"]

                bsz = imgs.size(0)
                real_labels = torch.full((bsz, 1), self.real_label_value, device=self.device)
                fake_labels = torch.full((bsz, 1), self.fake_label_value, device=self.device)

                texts = []
                for cap, cls in zip(caps, classes):
                    if isinstance(cap, bytes):
                        cap = cap.decode("utf-8", errors="ignore")
                    if isinstance(cls, bytes):
                        cls = cls.decode("utf-8", errors="ignore")
                    cap = (cap or "").strip()
                    cname = CLASS_NAME_MAP.get(str(cls), "").strip()
                    if cname and random.random() < self.class_mix_prob:
                        txt = f"a {cname} flower. {cap}" if cap else f"a {cname} flower"
                    else:
                        txt = cap if cap else (f"a {cname} flower" if cname else "a flower")
                    texts.append(txt)

                text_features = self.encode_text(texts)

                # D step
                self.netD.zero_grad(set_to_none=True)
                real_in = diffaugment(imgs) if self.use_diffaug else imgs
                out_real = self.netD(real_in, text_features)
                d_real = self.criterion(out_real, real_labels)

                z = torch.randn(bsz, self.z_dim, device=self.device) * self.z_trunc
                with torch.no_grad():
                    fake_imgs = self.netG(z, text_features)
                fake_in = diffaugment(fake_imgs.detach()) if self.use_diffaug else fake_imgs.detach()
                out_fake = self.netD(fake_in, text_features)
                d_fake = self.criterion(out_fake, fake_labels)

                d_loss = d_real + d_fake
                d_loss.backward()
                self.optD.step()

                # G step
                self.netG.zero_grad(set_to_none=True)
                z = torch.randn(bsz, self.z_dim, device=self.device) * self.z_trunc
                fake_imgs = self.netG(z, text_features)
                fake_in_g = diffaugment(fake_imgs) if self.use_diffaug else fake_imgs
                out_fake_for_g = self.netD(fake_in_g, text_features)

                g_adv = self.criterion(out_fake_for_g, real_labels)
                c_loss = self.clip_image_text_loss(fake_imgs, text_features)
                g_total = g_adv + lam_clip * c_loss
                g_total.backward()
                self.optG.step()

                self.ema.update(self.netG)

                sum_d += float(d_loss.item())
                sum_gadv += float(g_adv.item())
                sum_clip += float(c_loss.item())
                sum_gtot += float(g_total.item())
                steps += 1

                pbar.set_postfix({
                    "D": f"{d_loss.item():.3f}",
                    "G_adv": f"{g_adv.item():.3f}",
                    "CLIP": f"{c_loss.item():.3f}",
                    "G_tot": f"{g_total.item():.3f}"
                })

            with open(self.train_log_csv, "a", newline="") as fp:
                w = csv.writer(fp)
                w.writerow([
                    epoch,
                    sum_d / max(1, steps),
                    sum_gadv / max(1, steps),
                    sum_clip / max(1, steps),
                    sum_gtot / max(1, steps),
                    steps
                ])

            if (epoch % self.save_every) == 0:
                self._save_checkpoint(epoch)
