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
    #y_d = round((y_d + 1)/2 * h) 
    #x_d = round((x_d + 1)/2 * w)
    #if y_d <= h-1 and x_d <= w-1:
    #    return y_d,x_d
    #else:
    #    return y,x
    return y_d,x_d

def distorting_img(img_path):
    img = np.array(Image.open(img_path)).astype(np.float32)
    print('img.shape:',img.shape)
    print('img.min:',img.min())
    print('img.max:',img.max())
    h,w,_ = img.shape
    img_d = np.zeros(img.shape)
    grid = torch.zeros(h,w,2)
    for y in range(h):
        for x in range(w):
            #img_d[distorted_ordinates(y,x,h,w)] = img[y,x]
            grid[y][x][1],grid[y][x][0] = distorted_ordinates(y,x,h,w)
    grid = grid.unsqueeze(0)
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    img_d = functional.grid_sample(img,grid,mode='bilinear',padding_mode='zeros')
    
    #for y in range(1,h-1):
    #    for x in range(1,w-1):
    #        if img_d[y,x].sum() == 0:
    #            img_d[y,x] = 0.25 * img_d[y+1,x] + img_d[y-1,x] + img_d[y,x+1] +img_d[y,x-1]
    
    img_d = img_d.squeeze().permute(1,2,0).detach().cpu().numpy()
    Image.fromarray(img_d.astype(np.uint8)).save('img_d6.png')

   
if __name__ == '__main__':
    distorting_img(img_path)
