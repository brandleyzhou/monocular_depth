'''This is a script for make3d evaluation, original resolution of RGB is 2272 x 1704,
    gt is 55 x 305. For monocular depth where 192 x 640. we take a ratio 1 : 3 = H : W
'''
import cv2
import numpy as np
import scipy.io
import os
import torch
import networks
import argparse
from layers import disp_to_depth
parser = argparse.ArgumentParser(
    description='Simple testing funtion for Monodepthv2 models.')
parser.add_argument('--model_folder',type = str,
                    help='the folder name of model')
parser.add_argument('--model_name',type = str)
parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
main_path = 'data_path/Make3d'

## load mdoel and trained weights
model_path = os.path.join(args.model_folder, args.model_name)
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False,plan = args.model_folder[35:40])
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4),nonlin=args.model_folder[41:45])

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()

def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log


test_filenames = os.listdir(os.path.join(main_path,'Test134'))

depths_gt = []
images = []
ratio = 2

h_ratio = 1 / (1.33333 * ratio)
color_new_height = 192
color_new_width = 640
depth_new_height = 21
for filename in test_filenames:
    mat = scipy.io.loadmat(os.path.join(main_path,'Gridlaserdata','depth_sph_corr-{}.mat'.format(filename[4:-4])))
    depths_gt.append(mat['Position3DGrid'][:,:,3])
    image = cv2.imread(os.path.join(main_path,'Test134','{}'.format(filename)))
    image = image[(2272 - color_new_height)//2:(2272 + color_new_height)//2,(1704 - color_new_width) // 2:(1704 + color_new_width)//2,:]
    images.append(image[:,:,::-1])
    #cv2.imwrite(os.path.join(main_path,'Test134_cropped','{}'.format(filename)),image)
depths_gt_resized = map(lambda x:cv2.resize(x,(305,407),interpolation=cv2.INTER_NEAREST),depths_gt)
depths_gt_cropped = list(map(lambda x:x[(55-21)//2:(55+21)//2, (305-21*3)//2:(305+21*3)//2],depths_gt))
errors = []

for i in range(len(test_filenames)):
    depth_gt = depths_gt_cropped[i]
    image = torch.from_numpy(images[i].astype(np.float64).copy())
    # predicting depths
    original_width, original_height = depth_gt.shape
    input_image = image.permute((2,0,1)).unsqueeze(0)/255
    input_image = input_image.to(device)
    features = encoder(input_image.float())
    outputs = depth_decoder(features)
    disp = outputs[("disp", 0)].squeeze(0)
    scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
    depth_pred = scaled_disp.detach().cpu().numpy()[0]
    depth_pred = cv2.resize(depth_pred,depth_gt.shape[::-1],interpolation = cv2.INTER_NEAREST)
    #depth_pred = depth_pred[(192-21)//2:(192+21)//2,(640-63)//2,(640+63)//2]
    mask = np.logical_and(depth_gt > 0,depth_gt < 70)
    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= np.median(depth_gt) / np.median(depth_pred)
    depth_pred[depth_pred > 70] = 70
    errors.append(compute_errors(depth_gt,depth_pred))
mean_error = np.mean(errors,0)
print(('{:>8} |' * 4).format('abs_rel','sq_rel','rmse','rmse_log'))
print(('{: 8.3f},'*4).format(*mean_error.tolist()))
