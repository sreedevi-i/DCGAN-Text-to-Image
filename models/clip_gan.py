# models/clip_gan.py
import torch
import torch.nn as nn

def conv_block(in_c, out_c, k, s, p, bn=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_c, out_c, k, s, p, bn=True):
    layers = [nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_dim=100, cond_dim=512, image_size=64, gf_dim=64):
        super().__init__()
        self.cond_fc = nn.Sequential(nn.Linear(cond_dim, 128), nn.ReLU(True))
        in_code = z_dim + 128
        self.fc = nn.Linear(in_code, gf_dim*8*4*4)
        self.up1 = deconv_block(gf_dim*8, gf_dim*4, 4, 2, 1)
        self.up2 = deconv_block(gf_dim*4, gf_dim*2, 4, 2, 1)
        self.up3 = deconv_block(gf_dim*2, gf_dim,   4, 2, 1)
        self.up4 = nn.ConvTranspose2d(gf_dim, 3, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, z, cond):
        c = self.cond_fc(cond)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        x = self.up4(x)
        return self.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, cond_dim=512, image_size=64, df_dim=64):
        super().__init__()
        self.down1 = conv_block(3, df_dim,   4, 2, 1, bn=False)
        self.down2 = conv_block(df_dim, df_dim*2, 4, 2, 1)
        self.down3 = conv_block(df_dim*2, df_dim*4, 4, 2, 1)
        self.down4 = conv_block(df_dim*4, df_dim*8, 4, 2, 1)
        self.cond_fc = nn.Linear(cond_dim, df_dim*8)
        self.final = nn.Conv2d(df_dim*8, 1, 4, 1, 0)

    def forward(self, img, cond):
        h = self.down1(img); h = self.down2(h); h = self.down3(h); h = self.down4(h)
        c = self.cond_fc(cond).unsqueeze(-1).unsqueeze(-1)
        c = c.repeat(1, 1, h.size(2), h.size(3))
        h = h + c
        out = self.final(h)
        return out.view(-1, 1)
