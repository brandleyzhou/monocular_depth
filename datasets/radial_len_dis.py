import numpy as np
import PIL.Image as Image
import torch.nn.functional as functional
import torch

img_path = 'cups.png' 

def apply_distortion(y,x, k1, k2):
    r2 = y**2 + x**2
    f = 1 + k1 * k2 + k2 * r2**2
    return y * f, x * f

def distorted_ordinates(y, x, h, w):
    y_n = 2 * (y/h) - 1
    x_n = 2 * (x/w) - 1
    # distortion function
    k1, k2 = 0.05,0.1
    y_d, x_d = apply_distortion(y_n,x_n,k1,k2) 
    return y_d,x_d

def distorting_img(img_path):
    img = np.array(Image.open(img_path)).astype(np.float32)
    h,w,_ = img.shape
    img_d = np.zeros(img.shape)
    grid = torch.zeros(h,w,2)
    for y in range(h):
        for x in range(w):
            grid[y][x][1],grid[y][x][0] = distorted_ordinates(y,x,h,w)
    grid = grid.unsqueeze(0)
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    img_d = functional.grid_sample(img,grid,mode='bilinear',padding_mode='zeros')
    img_d = img_d.squeeze().permute(1,2,0).detach().cpu().numpy()
    Image.fromarray(img_d.astype(np.uint8)).save('img_d6.png')

   
if __name__ == '__main__':
    distorting_img(img_path)
