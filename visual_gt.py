from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.cm as cm
from layers import disp_to_depth
from velo_inter import lin_interp
from tqdm import tqdm 
import argparse

# Options Arguments
parser = argparse.ArgumentParser(description = 'options for visualizing ground truth')
parser.add_argument('--date',
                type = str,
                default = '2011_10_03',
                choices = ['2011_10_03','2011_09_30','2011_09_29','2011_09_28','2011_09_26'])
parser.add_argument('--drive',
                type = str,
                help = 'which drive')
parser.add_argument('--length',
                type = int,
                default = 50,
                help = 'how many frames converted')
args = parser.parse_args()

try:
    folder_path = os.path.join('data_path/train','{}_drive_00{}_sync'.format(args.date,args.drive),'proj_depth/groundtruth','image_03')
except:
    folder_path = os.path.join('data_path/train','{}_drive_00{}_sync'.format(args.date,args.drive),'proj_depth/groundtruth','image_02')

print('Converting velo images to color maps from {}'.format(folder_path))
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

def get_value_range(img):
    max_min = []
    max_min.append(img.min())
    vmax = np.percentile(img,95)
    max_min.append(vmax)
    return max_min

def save_gt_visualization(gt,i):
    #  generating color maps and save them
    gt = (gt - gt.min())/(gt.max() - gt.min())
    max_min_gt = get_value_range(gt)
    normalizer_gt = mpl.colors.Normalize(vmin = max_min_gt[0], vmax = max_min_gt[1])
    mapper_gt = cm.ScalarMappable(norm=normalizer_gt, cmap='magma')
    colormapped_gt = (mapper_gt.to_rgba(gt)[:, :,:3] * 255).astype(np.uint8)
    im_gt = Image.fromarray(colormapped_gt)
    gt_path = os.path.join(os.path.dirname(__file__),'error_vis','{}gt.jpeg'.format(i))
    im_gt.save(gt_path)

def main(args):
    len_frames = args.length
    frames = os.listdir(folder_path)
    for i,frame in tqdm(enumerate(frames)):
        velo_path = os.path.join(folder_path,frame)
        depth_img = np.asarray(Image.open(velo_path))/ 255
        x,y = np.where(depth_img > 0)
        d = depth_img[depth_img != 0]
        xyd = np.stack((y,x,d)).T
        gt_img = lin_interp(depth_img.shape,xyd)
        save_gt_visualization(gt_img,i)
    print('compeltion')
if __name__ == "__main__":
    main(args)
