# tools/eval_zero_shot_class.py
# --- path guard ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------
import os, csv, json, argparse, numpy as np, torch
from tqdm import tqdm
import clip
import torch.nn.functional as F

from tools.common import set_seed, load_cfg, load_clip_model, load_generator, z_list, generate_images

def load_class_map(path):
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    if path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("class_names", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return getattr(mod, "CLASS_NAME_MAP")
    raise ValueError("Provide .json or .py with CLASS_NAME_MAP")

@torch.no_grad()
def clip_zeroshot_classifier(clip_model, device, classnames, templates):
    zeroshot_weights = []
    for cname in classnames:
        texts = [t.format(cname) for t in templates]
        tokens = clip.tokenize(texts, truncate=True).to(device)
        class_emb = clip_model.encode_text(tokens).float()
        class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)
        class_emb = class_emb.mean(dim=0)
        class_emb = class_emb / class_emb.norm()
        zeroshot_weights.append(class_emb)
    return torch.stack(zeroshot_weights, dim=1).to(device)  # [D, K]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--class_map_file", required=True)  # points to CLASS_NAME_MAP
    ap.add_argument("--n_per_class", type=int, default=8)
    ap.add_argument("--trunc", type=float, default=1.0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ema_ckpt", default=None)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    clip_model = load_clip_model(device)
    G = load_generator(args.gen_ckpt, cfg, device, ema_ckpt=args.ema_ckpt)

    cmap = load_class_map(args.class_map_file)  # dict: "class_00001" -> "daisy"
    class_ids = sorted(cmap.keys())
    classnames = [cmap[cid] for cid in class_ids]

    templates = [
        "a photo of a {} flower",
        "a close-up of a {}",
        "a {} bloom",
        "a macro shot of a {} flower"
    ]
    W = clip_zeroshot_classifier(clip_model, device, classnames, templates)  # [D,K]

    prompts = [f"a {name} flower" for name in classnames]
    total = args.n_per_class * len(prompts)
    z = z_list(total, cfg.Z_DIM, seed=args.seed)
    pil = generate_images(G, clip_model, device, prompts, args.n_per_class, z, trunc=args.trunc, image_size=cfg.IMAGE_SIZE)

    import torchvision.transforms as T
    tr = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.48145466,0.4578275,0.40821073),
                                (0.26862954,0.26130258,0.27577711))])

    X = torch.stack([tr(im.convert("RGB")) for im in pil]).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(X).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)  # [N,D]
        logits = feats @ W  # [N,K]
        preds = logits.argmax(dim=1).cpu().numpy()

    N = len(pil)
    y = np.repeat(np.arange(len(classnames)), args.n_per_class)
    acc = float((preds == y).mean())

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","top1_acc","K","N","n_per_class"])
        from tools.common import parse_epoch_from_ckpt
        e = parse_epoch_from_ckpt(args.gen_ckpt)
        w.writerow([e, f"{acc:.6f}", len(classnames), N, args.n_per_class])

if __name__ == "__main__":
    main()
