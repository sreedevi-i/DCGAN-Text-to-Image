# txt2image_dataset.py
import io
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class Text2ImageDataset(Dataset):
    def __init__(self, hdf5_path, split='train', image_size=64):
        self.hdf5_path = hdf5_path
        self.split = split
        self.image_size = image_size
        self._open()

    def _open(self):
        self.h5 = h5py.File(self.hdf5_path, 'r', swmr=True)
        self.grp = self.h5[self.split]
        self.keys = list(self.grp.keys())

    def __len__(self):
        return len(self.keys)

    def _ensure_open(self):
        try:
            _ = self.h5.id
        except Exception:
            self._open()

    def __getitem__(self, idx):
        self._ensure_open()
        k = self.keys[idx]
        g = self.grp[k]
        img_bytes = bytes(g['img'][()])
        with Image.open(io.BytesIO(img_bytes)) as im:
            im = im.convert('RGB')
            im = im.resize((self.image_size, self.image_size), Image.BICUBIC)
        img = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        img = img * 2 - 1
        txt_raw = g['txt'][()]
        cls_raw = g['class'][()]
        txt = txt_raw.decode('utf-8') if isinstance(txt_raw, bytes) else str(txt_raw)
        cls_ = cls_raw.decode('utf-8') if isinstance(cls_raw, bytes) else str(cls_raw)
        return {"image": img, "caption": txt, "class": cls_}
