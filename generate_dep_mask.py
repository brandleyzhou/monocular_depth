import torch
from PIL import Image 
#from imageio import imread,imwrite
import numpy as np
def generate_depth_mask(img1,img2,threshold):
    img1_avg = torch.mean(img1,2,True)
    img2_avg = torch.mean(img2,2,True)
    mask = torch.abs(img1_avg - img2_avg) < threshold
    print(type(mask))
    print(type(mask[0][0][0]))
    return mask

def imread(path):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
a = imread('assets/0000000057.png')
c = imread("assets/0000000059.png")
b = imread("assets/0000000058.png")
a = torch.from_numpy(np.asarray(a)).float() 
b = torch.from_numpy(np.asarray(b)).float()
c = torch.from_numpy(np.asarray(c)).float()
print(a.max(),a.min())
mask = generate_depth_mask(c,b,10).cpu().numpy()
print(mask.sum())
print(mask.max(),mask.min())
print(type(mask[0][0][0]))
mask = Image.fromarray(255 - mask[:,:,0] * 255)
mask.save("assets/depth_mask.png")
