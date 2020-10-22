from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import datasets
import options
from layers import SSIM
from encoder import Encoder
from decoder import Decoder
from tqdm import tqdm
import os
from utils import *
device = 'cuda'
options = options.MonodepthOptions()
opt = options.parse()

datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                 "kitti_odom": datasets.KITTIOdomDataset}
dataset = datasets_dict[opt.dataset]

fpath = os.path.join('../..', "splits", opt.split, "{}_files.txt")

train_filenames = readlines(fpath.format("train"))
val_filenames = readlines(fpath.format("val"))
val_filenames = val_filenames[:12]
img_ext = '.png' if opt.png else '.jpg'
global num_train_samples
num_train_samples = len(train_filenames)
num_total_steps = num_train_samples // opt.batch_size * opt.num_epochs
        
train_dataset = dataset(
    opt.data_path, train_filenames, opt.height, opt.width,
    opt.frame_ids, 4, is_train=True, img_ext=img_ext,data_augment=opt.data_augment,apply_distortion = opt.apply_distortion)
train_loader = DataLoader(
    train_dataset, opt.batch_size, True,
    num_workers=opt.num_workers, pin_memory=True, drop_last=True)
        
val_dataset = dataset(
    opt.data_path, val_filenames, opt.height, opt.width,
    opt.frame_ids, 4, is_train=False, img_ext=img_ext,data_augment=opt.data_augment)
val_loader = DataLoader(
    val_dataset, opt.batch_size,shuffle=False,
    num_workers=opt.num_workers, pin_memory=True, drop_last=True)
val_iter = iter(val_loader)

models={}
parameters_to_learn=[]
models["encoder"] = Encoder(18, None).to(device)
parameters_to_learn += list(models["encoder"].parameters())
models["decoder"] = Decoder(models["encoder"].num_ch_enc).to(device)
parameters_to_learn += list(models["decoder"].parameters())

optimizer = optim.Adam(parameters_to_learn,opt.learning_rate) 
model_lr_scheduler = optim.lr_scheduler.StepLR(
                    optimizer,opt.scheduler_step_size, 0.1)#defualt = 15'step size of the scheduler'


def val():
    """Validate the model on a single minibatch
    """
    set_eval()
    i = 0
    try:
        inputs = val_iter.next()
    except StopIteration:
        val_iter = iter(val_loader)
        inputs = val_iter.next()

    with torch.no_grad():
        if i == 0:
            outputs, losses = process_batch(inputs)
            i += 1
        else:
            outputs, losses = process_batch(inputs)
        #if "depth_gt" in inputs:
        #    compute_depth_losses(inputs, outputs, losses)
        del inputs, outputs, losses
        set_train()

def set_eval():
    """Convert all models to testing/evaluation mode
    """
    for m in models.values():
        m.eval()

def set_train():
    """Convert all models to training mode
    """
    for m in models.values():
        m.train()

'''
feature loss function
'''
ssim = SSIM()
def get_smooth_loss(disp, img):
    b, _, h, w = disp.size()
    img = F.interpolate(img, (h, w), mode='area')

    disp_dx, disp_dy = gradient(disp)
    img_dx, img_dy = gradient(img)

    disp_dxx, disp_dxy = gradient(disp_dx)
    disp_dyx, disp_dyy = gradient(disp_dy)

    img_dxx, img_dxy = gradient(img_dx)
    img_dyx, img_dyy = gradient(img_dy)

    smooth1 = torch.mean(disp_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
              torch.mean(disp_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

    smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
              torch.mean(disp_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
              torch.mean(disp_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
              torch.mean(disp_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

    return 1 * smooth1+ 1 * smooth2
    #return -opt.dis * smooth1+ opt.cvt * smooth2

def gradient(D):
    dy = D[:, :, 1:] - D[:, :, :-1]
    dx = D[:, :, :, 1:] - D[:, :, :, :-1]
    return dx, dy

def robust_l1(pred, target):
    eps = 1e-3
    return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

def compute_reprojection_loss(pred, target):
    photometric_loss = robust_l1(pred, target).mean(1, True)
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
    return reprojection_loss

def compute_losses(inputs, outputs, features):
    loss_dict = {}
    interval = 1000
    target = inputs[("color", 0, 0)]
    for i in range(5):
        f=features[i]
        smooth_loss = get_smooth_loss(f, target)
        loss_dict[('smooth_loss', i)] = smooth_loss/ (2 ** i)/5

    for scale in opt.scales:
        """
        initialization
        """
        pred = outputs[("disp", 0, scale)]

        _,_,h,w = pred.size()
        target = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
        min_reconstruct_loss = compute_reprojection_loss(pred, target)
        loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean()/len(opt.scales)

#        if self.count % interval == 0:
#            img_path = os.path.join('/node01_data5/monodepth2-test/odo', 'auto_{:0>4d}_{}.png'.format(self.count // interval, scale))
#            plt.imsave(img_path, pred[0].transpose(0,1).transpose(1,2).data.cpu().numpy())
#            img_path = os.path.join('/node01_data5/monodepth2-test/odo', 'img_{:0>4d}_{}.png'.format(self.count // interval, scale))
#            plt.imsave(img_path, target[0].transpose(0, 1).transpose(1, 2).data.cpu().numpy())
#
#    self.count += 1
    return loss_dict

########################################################################
for epoch in range(opt.num_epochs):
    set_train()
    for batch_idx, inputs in enumerate(tqdm(train_loader)):
        for key,item in inputs.items():
            inputs[key] = item.to(device)
        features = models["encoder"](inputs[("color",0,0)])
        outputs = models["decoder"](features,0)
        loss_dict = compute_losses(inputs, outputs, features)
        losses = 0
        for scale in opt.scales:
            losses += loss_dict[('min_reconstruct_loss',scale)]
        '''first time we jsut takes scale = 3min_reconstruct_loss into account '''
        #for i in range(5):
        #   losses += loss_dict[('smooth_loss',i)]
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        early_phase = batch_idx == 0
        late_phase = batch_idx == num_train_samples//opt.batch_size - 1
        #late_phase = batch_idx == 3316
        #if early_phase or late_phase:
        #    val()
    model_lr_scheduler.step()
    save_path = "../../models/autoencoder_{}.pth".format(epoch)   
    torch.save(models["encoder"].state_dict(),save_path)

