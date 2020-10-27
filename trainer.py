from __future__ import absolute_import, division, print_function
#from torchsummary import summary
from tqdm import tqdm
import numpy as np
import time
import PIL
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch.nn import DataParallel as DDP
#from torchvision.utils import save_image
#from toch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
    #from tensorboardX import SummaryWriter
except:
    from tensorboardX import SummaryWriter
    #from torch.utils.tensorboard import SummaryWriter
import json

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks
from networks.layers import SSIM,Backproject,Project
from IPython import embed

##feature extractor building
#def build_extractor(num_layers,pretrained_path):
#    '''maybe using resnet-18 can get better results'''
#    extractor = networks.mono_autoencoder.encoder.Encoder(50, pretrained_path)
#    extractor.load_state_dict(torch.load(pretrained_path))
#    for param in extractor.parameters():
#        param.requires_grad = False
#    return extractor

def build_extractor(num_layers,pretrained_path):
    extractor = networks.mono_autoencoder.encoder.Encoder(50,pretrained_path)
    checkpoint = torch.load(pretrained_path, map_location = 'cpu')
    for name, param in extractor.state_dict().items():
        extractor.state_dict()[name].copy_(checkpoint['state_dict']['Encoder.'+name])
    for param in extractor.parameters():
        param.requires_grad = False
    return extractor

def get_value_range(img):
    max_min = []
    max_min.append(img.min())
    vmax = np.percentile(img,95)
    max_min.append(vmax)
    return max_min

#def save_error_visualization(ssmi,target,pred,L1):
#    L1 = L1[0,:,:].squeeze(0).squeeze(0).cpu()
#    target = target[0,:,:,:].squeeze(0).cpu()
#    pred = pred[0,:,:,:].squeeze(0).cpu()
#    error = ssmi[0,:,:].squeeze(0).squeeze(0).cpu()
#    ## save pred and target
#    #pred = (pred.permute(1,2,0) * 255).numpy()
#    #pred = pil.fromarray(pred)
#    #target = (target.permute(1,2,0) * 255).numpy()
#    #target = pil.fromarray(target)
#    ## ssmi visualization
#    max_min_diff = get_value_range(error)
#    normalizer_diff = mpl.colors.Normalize(vmin = max_min_diff[0], vmax = max_min_diff[1])
#    mapper = cm.ScalarMappable(norm=normalizer_diff, cmap='magma')
#    error = (mapper.to_rgba(error)[:, :,:3] * 255).astype(np.uint8)
#    error = pil.fromarray(error)
#    ## l1 visualization
#    max_min_diff = get_value_range(L1)
#    normalizer_diff = mpl.colors.Normalize(vmin = max_min_diff[0], vmax = max_min_diff[1])
#    mapper = cm.ScalarMappable(norm=normalizer_diff, cmap='magma')
#    L1 = (mapper.to_rgba(L1)[:, :,:3] * 255).astype(np.uint8)
#    L1 = pil.fromarray(L1)
#    ## create paths
#    error_path = os.path.join('.','error_vis','{}_error.jpeg'.format(time.ctime(time.time())))
#    L1_path = os.path.join('.','error_vis','{}_L1.jpeg'.format(time.ctime(time.time())))
#    target_path = os.path.join('.','error_vis','{}_target.jpeg'.format(time.ctime(time.time())))
#    pred_path = os.path.join('.','error_vis','{}_pred.jpeg'.format(time.ctime(time.time())))
#    ## save images
#    error.save(error_path)
#    L1.save(L1_path)
#    save_image(target,target_path)
#    save_image(pred,pred_path)

def generate_depth_mask(img1,img2,threshold):
    img1_avg = torch.mean(img1,1,True)
    img2_avg = torch.mean(img2,1,True)
    mask = torch.abs(img1_avg - img2_avg) < threshold
    try:
        mask.to(torch.device("cuda:0"))
    except:
        mask.to(torch.device("cuda:1"))
    return mask.float()

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_dir = os.path.join(os.path.expanduser("~"),"{}/{}_{}_checkpoints_{}_v{}_{}".format(self.opt.log_dir,self.opt.plan,self.opt.acti_func,self.opt.data_size,self.opt.using_v,self.opt.data_augment))
        self.log_path = os.path.join(self.log_dir, self.opt.model_name)
        # if using new architecture ,model will be saved in a dir ending with new
        if self.opt.add_neighboring_frames == 1:
            self.neighboring_depth = {}
            self.log_dir = os.path.join(os.path.expanduser("~"),"{}/{}_{}_checkpoints_{}_v{}_new_{}".format(self.opt.log_dir,self.opt.plan,self.opt.acti_func,self.opt.data_size,self.opt.using_v,self.opt.data_augment))
            self.log_path = os.path.join(self.log_dir, self.opt.model_name)
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        #self.device = torch.device("cuda:1" if self.opt.ada else "cuda:0")#not using cuda?
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")#not using cuda?
        self.num_scales = len(self.opt.scales)#scales = [0,1,2,3]'scales used in the loss'
        self.num_input_frames = len(self.opt.frame_ids)#frames = [0,-1,1]'frame to load'
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        #defualt is pose_model_input = 'pairs'

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        ## add feature extractor module:
        #self.models["extractor"] = build_extractor(
        #        50, self.opt.extractor_pretrained_path).to(self.device)
        self.models["extractor"] = build_extractor(
                self.opt.num_layers, self.opt.extractor_pretrained_path).to(self.device)
        
        self.models["encoder"] = networks.mono_autoencoder.encoder.Encoder(
                50, 'gpfs/home/mxa19ypu/.cache/torch/checkpoints/resnet50-19c8e357')
        #self.models["encoder"] = networks.ResnetEncoder(
        #    self.opt.num_layers, self.opt.weights_init == "pretrained",plan = self.opt.plan)
        #defualt = 18 choice=[18,34,50,101,152]
        para_sum = sum(p.numel() for p in self.models['encoder'].parameters())
        print(para_sum)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales,nonlin=self.opt.acti_func,using_v=self.opt.using_v)
        print(self.device)
        self.models["encoder"].to(self.device)
        #summary(self.models["depth"])
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:#use_pose_net = True
            if self.opt.pose_model_type == "separate_resnet":#defualt=separate_resnet  choice = ['normal or shared']
                #seperate means pose_encoder using a different ResnetEncoder
                #shared means pose_encoder using a same ResnetEncoder as depth_encoder
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    #self.opt.num_layers,
                    18,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames,plan = self.opt.plan)#num_input_images=2
                #num_input_frames = 2 different from model['encoder']
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            #self.models["pose_encoder"] = DDP(self.models["pose_encoder"])
            self.models["pose_encoder"].to(self.device)
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
            #self.parameters_to_train contains learnable parameters of 4 models
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.MaskDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1),mask_name='predictive_mask')
            
            print('\nUsing predictive mask\n')
            
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        
        # using distance constraint mask instead of a scalar
        if self.opt.distance_constraint_mask and self.opt.add_neighboring_frames:
            print('\nUsing distance_constraint_mask\n')
            #taking three frames into mask encoder
            self.models["distance_constraint_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images= 3 ,plan = self.opt.plan)
            self.models['distance_constraint_encoder'].to(self.device)
            self.parameters_to_train += list(self.models['distance_constraint_encoder'].parameters())
            #decode and get distance constraint mask
            self.models['distance_constraint_mask'] = networks.MaskDecoder(
                    self.models['distance_constraint_encoder'].num_ch_enc,self.opt.scales,
                    num_output_channels = 1,mask_name = 'distance_constraint_mask')
            self.models['distance_constraint_mask'].to(self.device)
            self.parameters_to_train += list(self.models['distance_constraint_mask'].parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)#learning_rate=1e-4
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)#defualt = 15'step size of the scheduler'

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        val_filenames = val_filenames[:12]
        img_ext = '.png' if self.opt.png else '.jpg'
        #change dataset_size
        if self.opt.data_size == "part":
            train_filenames = train_filenames[:3000]
            val_filenames = val_filenames[:300]
        global num_train_samples
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        #batch_size = 12,epoch_number = 20
        #train_dataset = self.dataset(
        #    self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
        #    self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
                
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext,data_augment=self.opt.data_augment,apply_distortion = self.opt.apply_distortion)
        
        
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
                
        #val_dataset = self.dataset(
        #    self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
        #    self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext,data_augment=self.opt.data_augment)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size,shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)#defualt=[0,1,2,3]'scales used in the loss'
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)#in layers.py
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.init_time = time.time()
        start_time = time.ctime(self.init_time)
        print('Training starts at {}\n'.format(start_time))
        if isinstance(self.opt.load_weights_folder,str):
            if self.opt.load_weights_folder[-2] == "1":
                self.epoch_start = int(self.opt.load_weights_folder[-2:]) + 1
            else:
                self.epoch_start = int(self.opt.load_weights_folder[-1]) + 1
        else:
            self.epoch_start = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs - self.epoch_start):
            self.epoch = self.epoch_start + self.epoch 
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:#number of epochs between each save defualt =1
                self.save_model()
        self.total_training_time = time.time() - self.init_time
        completion_time = time.ctime(time.time())
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))
        print('Training ends at {}'.format(completion_time))
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.every_epoch_start_time = time.time()
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(tqdm(self.train_loader)):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs,save_error = False)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx == 0
            late_phase = batch_idx == num_train_samples//self.opt.batch_size - 1
            #late_phase = batch_idx == 3316
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
        self.every_epoch_end_time = time.time()
        self.model_lr_scheduler.step()
        if self.epoch == self.epoch_start: 
            self.second_of_first_epoch = self.every_epoch_end_time-self.every_epoch_start_time
            the_second_of_arrival = (self.opt.num_epochs - self.epoch_start - 1) * self.second_of_first_epoch + time.time()
            self.the_time_of_arrival = time.ctime(the_second_of_arrival)
        print("====>training time of this epoch:{} |xxxxx| the Time Of Arrival:{} ".format(sec_to_hm_str(self.every_epoch_end_time-self.every_epoch_start_time),self.the_time_of_arrival))

    def process_batch(self, inputs,save_error = False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():#inputs.values() has :12x3x196x640.
            inputs[key] = ipt.to(self.device)#put tensor in gpu memory
        if self.opt.distance_constraint_mask:
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids],1)
            all_features = self.models["distance_constraint_encoder"](all_color_aug)#stacked frames processing color together
            masks = self.models['distance_constraint_mask'](all_features)
            self.distance_constraint_mask = masks[("distance_constraint_mask",0)]
        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)#stacked frames processing color together
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]#? what does inputs mean?

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)
            # using new architecture
            if self.opt.add_neighboring_frames == 1:
                self.depth_mask = []
                feature_previous = self.models["encoder"](inputs["color_aug",-1,0])
                feature_next =  self.models["encoder"](inputs["color_aug",1,0])
                self.depth_mask.append(generate_depth_mask(inputs[("color_aug",0,0)],inputs[("color_aug",-1,0)],self.opt.threshold))
                self.depth_mask.append(generate_depth_mask(inputs[("color_aug",0,0)],inputs[("color_aug",1,0)],self.opt.threshold))
                outputs_previous = self.models["depth"](feature_previous)
                outputs_next = self.models["depth"](feature_next)
                for scale in self.opt.scales:    
                    #generate two sets of  depth maps
                    disp_previous = F.interpolate(outputs_previous[("disp",scale)],[self.opt.height,self.opt.width],mode="bilinear",align_corners=False)
                    disp_next = F.interpolate(outputs_next[("disp",scale)],[self.opt.height,self.opt.width],mode="bilinear",align_corners=False)
                    _,self.neighboring_depth[("depth_previous",scale)] = disp_to_depth(disp_previous,self.opt.min_depth,self.opt.max_depth)
                    _,self.neighboring_depth[("depth_next",scale)] = disp_to_depth(disp_next,self.opt.min_depth,self.opt.max_depth)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
            #different form 1:*:* depth maps ,it will output 2:*:* mask maps

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        ## add feature loss
        self.generate_feature_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, save_error = save_error)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            #pose_feats is a dict:
            #key:
            """"keys
                0
                -1
                1
            """
            for f_i in self.opt.frame_ids[1:]:
                #frame_ids = [0,-1,1]
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]#nerboring frames
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    #axisangle and translation are two 2*1*3 matrix
                    #f_i=-1,1
                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        i = 0
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            if i == 0:
                outputs, losses = self.process_batch(inputs,save_error = False)
                i += 1
            else:
                outputs, losses = self.process_batch(inputs,save_error = False)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


#### adding features 
    def generate_feature_pred(self, inputs, outputs):
        outputs[("disp",0,0)] = outputs[("disp",0)]
        disp = outputs[("disp", 0, 0)]
        disp = F.interpolate(disp, [int(self.opt.height/2), int(self.opt.width/2)], mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            backproject = Backproject(self.opt.batch_size, int(self.opt.height/2), int(self.opt.width/2))
            project = Project(self.opt.batch_size, int(self.opt.height/2), int(self.opt.width/2))

            cam_points = backproject(depth, inputs[("inv_K",0)])
            pix_coords = project(cam_points, inputs[("K",0)], T)#[b,h,w,2]
            img = inputs[("color", frame_id, 0)]
            src_f = self.models["extractor"](img)[0]
            outputs[("feature", frame_id, 0)] = F.grid_sample(src_f, pix_coords, padding_mode="border")
        return outputs


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # without interpolate
                if self.opt.using_v not in [3,4]:
                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)#disp_to_depth function is in layers.py

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    
    def robust_11(self,pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps**2)

    def compute_perceptional_loss(self,tgt_f,src_f):
        perceptional_loss = self.robust_11(tgt_f,src_f).mean(1,True)
        return perceptional_loss

    def compute_reprojection_loss(self, pred, target, save_error=False):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        if save_error == True:
            save_error_visualization(ssim_loss,target,pred,l1_loss)
            print('visualized valid set')
        return reprojection_loss

    def compute_losses(self, inputs, outputs,save_error=False):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        losses['perceptional_loss'] = 0

        for scale in self.opt.scales:
            #scales=[0,1,2,3]
            loss = 0
            reprojection_losses = []
            perceptional_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            ##add feature map
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            
            #adding feature_loss
            for frame_id in self.opt.frame_ids[1:]:
                src_f = outputs[("feature", frame_id, 0)]
                tgt_f = self.models["extractor"](inputs[("color", 0, 0)])[0]
                perceptional_losses.append(self.compute_perceptional_loss(tgt_f, src_f))
            perceptional_loss = torch.cat(perceptional_losses, 1)

            min_perceptional_loss, outputs[("min_index", scale)] = torch.min(perceptional_loss, dim=1)
            losses[('min_perceptional_loss', scale)] = self.opt.perception_weight * min_perceptional_loss.mean() / len(self.opt.scales)
       
            losses['perceptional_loss'] += losses[('min_perceptional_loss',scale)]
            
            # photometric_loss
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target,save_error))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target,save_error))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
                    if self.opt.mask_plan in [1,2,3]:
                        self.distance_constraint_automask = identity_reprojection_loss.min(1,keepdim=True)

            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["predictive_mask", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask
            #using distance_constraint_mask
            #elif self.opt.distance_constraint_mask:
                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda()) if torch.cuda.is_available() else   0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cpu())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                    #identity_reprojection_loss.shape).cuda() * 0.00001
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda(1) * 0.00001 if self.opt.no_cuda else torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
                else:
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cpu() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        #total_loss = (1 - self.opt.perception_weight) * total_loss + self.opt.perception_weight * losses['perceptional_loss']
        total_loss = total_loss + self.opt.perception_weight * losses['perceptional_loss']
        
        #using new architecture
        if self.opt.add_neighboring_frames == 1:
            depth_loss_sum = 0
            depth_loss_weights_sum = 0
            if self.opt.depth_multiscale:
                for i in self.opt.scales:
                    #testing before
                    depth_mid = torch.abs(self.neighboring_depth[("depth_previous",i)] - \
                        self.neighboring_depth[("depth_next",i)]) / 2 if self.opt.respective_depth_constraint \
                        else  torch.abs(self.neighboring_depth[("depth_previous",i)] - \
                        self.neighboring_depth[("depth_next",i)]) / 2 + self.neighboring_depth[("depth_next",i)]
                    ## L2 loss
                    #depth_loss = nn.MSELoss()(torch.abs(self.neighboring_depth[("depth_previous",i)] - outputs[("depth",0,i)]), depth_mid) * self.depth_mask[0] + \
                    #        nn.MSELoss()(torch.abs(self.neighboring_depth[("depth_next",i)] - outputs[("depth",0,i)]), depth_mid)*self.depth_mask[1] if self.opt.respective_depth_constraint \
                    #        else nn.MSELoss()(depth_mid , outputs[("depth",0,i)])
                    
                    depth_loss = torch.abs(torch.abs(self.neighboring_depth[("depth_previous",i)] - outputs[("depth",0,i)]) - depth_mid) * self.depth_mask[0] + \
                            torch.abs(torch.abs(self.neighboring_depth[("depth_next",i)] - outputs[("depth",0,i)]) - depth_mid)*self.depth_mask[1] if self.opt.respective_depth_constraint \
                            else torch.abs(depth_mid - outputs[("depth",0,i)])
                    #depth_loss = torch.abs(torch.abs(self.neighboring_depth[("depth_previous",i)] - outputs[("depth",0,i)]) - depth_mid) + \
                    #        torch.abs(torch.abs(self.neighboring_depth[("depth_next",i)] - outputs[("depth",0,i)]) - depth_mid) if self.opt.respective_depth_constraint \
                    #        else torch.abs(depth_mid - outputs[("depth",0,i)])
                    
                    if self.opt.distance_constraint_mask:
                        depth_lossing =  self.opt.depth_loss_weight * (depth_loss * self.distance_constraint_mask).mean()
                        if not self.opt.disable_BCELoss:#when setting distance mask will doing this 
                            depth_loss_weights = self.opt.distance_mask_weight* nn.BCELoss()\
                                    (self.distance_constraint_mask, \
                                    torch.ones(self.distance_constraint_mask.shape).cuda()) \
                                    if torch.cuda.is_available() \
                                    else \
                                    self.opt.distance_mask_weight * nn.BCELoss()\
                                    (self.distance_constraint_mask, \
                                    torch.ones(self.distance_constraint_mask.shape).cpu())
                            depth_loss_weights_sum += depth_loss_weights
                            if float(depth_loss_weights)  == 0:
                                print("distance_mask is useless")
                    else:
                        if self.opt.mask_plan == 0:
                            depth_lossing = (depth_loss * self.opt.depth_loss_weight).mean()
                        elif self.opt.mask_plan == 1:
                            depth_lossing =  (depth_loss * self.distance_constraint_automask[0]).mean()
                        elif self.opt.mask_plan == 2:
                            depth_lossing = self.opt.depth_loss_weight * (depth_loss * self.distance_constraint_automask[0]).mean()
                        elif self.opt.mask_plan == 3:
                            depth_lossing = self.opt.depth_loss_weight * (depth_loss * self.distance_constraint_automask).mean()
                    depth_loss_sum += depth_lossing
            else:
                depth_mid = torch.abs(self.neighboring_depth[("depth_previous",0)] - \
                        self.neighboring_depth[("depth_next",0)]) / 2 if self.opt.respective_depth_constraint \
                        else  torch.abs(self.neighboring_depth[("depth_previous",0)] - \
                        self.neighboring_depth[("depth_next",0)]) / 2 + self.neighboring_depth[("depth_next",0)]
                for i in self.opt.scales:
                    ## L2 loss
                    #depth_loss = nn.MSELoss()(torch.abs(self.neighboring_depth[("depth_previous",0)] - outputs[("depth",0,i)]), depth_mid) * self.depth_mask[0] + \
                    #        nn.MSELoss()(torch.abs(self.neighboring_depth[("depth_next",0)] - outputs[("depth",0,i)]), depth_mid)*self.depth_mask[1] if self.opt.respective_depth_constraint \
                    #        else nn.MSELoss()(depth_mid, outputs[("depth",0,i)])
                    
                    depth_loss = torch.abs(torch.abs(self.neighboring_depth[("depth_previous",0)] - outputs[("depth",0,i)]) - depth_mid) * self.depth_mask[0] + \
                            torch.abs(torch.abs(self.neighboring_depth[("depth_next",0)] - outputs[("depth",0,i)]) - depth_mid)*self.depth_mask[1] if self.opt.respective_depth_constraint \
                            else torch.abs(depth_mid - outputs[("depth",0,i)])
                    #depth_loss = torch.abs(torch.abs(self.neighboring_depth[("depth_previous",0)] - outputs[("depth",0,i)]) - depth_mid) + \
                    #        torch.abs(torch.abs(self.neighboring_depth[("depth_next",0)] - outputs[("depth",0,i)]) - depth_mid) if self.opt.respective_depth_constraint\
                    #        else torch.abs(depth_mid - outputs[("depth",0,i)])
                    if self.opt.distance_constraint_mask:
                        depth_lossing =  self.opt.depth_loss_weight * (depth_loss * self.distance_constraint_mask).mean()
                        if not self.opt.disable_BCELoss:
                            depth_loss_weights = self.opt.distance_mask_weight* nn.BCELoss()\
                                    (self.distance_constraint_mask, \
                                    torch.ones(self.distance_constraint_mask.shape).cuda()) \
                                    if torch.cuda.is_available() \
                                    else \
                                    self.opt.distance_mask_weight * nn.BCELoss()\
                                    (self.distance_constraint_mask, \
                                    torch.ones(self.distance_constraint_mask.shape).cpu())
                            depth_loss_weights_sum += depth_loss_weights
                    else:
                        if self.opt.mask_plan == 0:
                            depth_lossing = (depth_loss * self.opt.depth_loss_weight).mean()
                        elif self.opt.mask_plan == 1:
                            depth_lossing =  (depth_loss * self.distance_constraint_automask[0]).mean()
                        elif self.opt.mask_plan == 2:
                            depth_lossing = self.opt.depth_loss_weight * (depth_loss * self.distance_constraint_automask[0]).mean()
                        elif self.opt.mask_plan == 3:
                            depth_lossing = self.opt.depth_loss_weight * (depth_loss * self.distance_constraint_automask).mean()
                    depth_loss_sum += depth_lossing
            depth_loss_sum /= 4
            if depth_loss_sum == 0:
                print("depth_loss is useless")
            depth_loss_weights_sum /= 4
            if self.opt.combined_loss == True:
                total_loss = (1-self.opt.depth_loss_weight) * total_loss + depth_loss_sum + depth_loss_weights_sum
            else:
                total_loss += depth_loss_sum + depth_loss_weights_sum
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        print_string = "epoch {:>3} | batch_idx {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} "
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("predictive_mask", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
