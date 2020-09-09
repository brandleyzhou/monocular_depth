import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

toten = ToTensor()
topil = ToPILImage()

def apply_distortion(pts,k1,k2):
    r2 = torch.pow(pts,2).sum(-1, keepdim=True)
    f = 1+ k1* r2 + k2 *r2* r2
    return pts * f

img = Image.open('cups.png')
ten = toten(img)
w,h = img.size

hv, wv = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])
pts = torch.stack([wv, hv], dim=-1)
pts_distorted = apply_distortion(pts, k1=0.5, k2=0.2)
out_distorted  = torch.nn.functional.grid_sample(ten[None, ...], pts_distorted[None, ...])
topil(out_distorted[0]).save('img_d.png')
