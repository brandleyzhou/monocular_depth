from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "data_path/raw_kitti"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default = "test_monodepth")
        
        self.parser.add_argument("--extractor_pretrained_path",
                                type=str,
                                help='path of pretrained feature extractor',
                                default = 'models/autoencoder.pth'
                                )
        self.parser.add_argument("--perception_weight",
                                type=float,
                                help='path of pretrained feature extractor',
                                default = 0.5
                                )
        
        # TRAINING options
        self.parser.add_argument("--data_augment",
                                 type = str,
                                 default = 'flipping',
                                 help = 'decide how to augment dataset',
                                 choices=["flipping","rotation"])
        self.parser.add_argument("--data_size",
                                 type = str,
                                 help = 'the size of training set and valid set',
                                 choices=["part","all"])
        self.parser.add_argument("--threshold",
                                 type = float,
                                 default = 0.04,
                                 help = 'threshold of photomatrix difference',
                                )
        self.parser.add_argument("--apply_distortion",
                                 type = bool,
                                 default = False,
                                 help = 'whether distorting images or not'
                                )
        self.parser.add_argument("--combined_loss",
                                 type = bool,
                                 default = False,
                                 help = 'how to weight loss elements',
                                )
        self.parser.add_argument("--depth_loss_weight",
                                 type = float,
                                 default = 0.0001,
                                 help = 'weight of depth loss',
                                )
        self.parser.add_argument("--mask_plan",
                                 type = int,
                                 default = 0,
                                 help = 'adding two neiboring frames into depth net',
                                 choices=[0,1,2,3])
        self.parser.add_argument("--add_neighboring_frames",
                                 type = int,
                                 default = 0,
                                 help = 'adding two neiboring frames into depth net',
                                 choices=[1,0])
        self.parser.add_argument("--plan",
                                 type = str,
                                 help = 'different type of encoder',
                                 choices = ["plan0","plan1","plan2","plan3","plan4","plan5"])
        self.parser.add_argument("--acti_func",
                                 type = str,
                                 help = ["sigm","relu"]
                                 )
        self.parser.add_argument("--using_v",
                                type = int,
                                default = 0,
                                choices = [0,1,2,3,4],
                                help = "decide which method  to decode disp-maps ----use my idea"
                                )
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mono_model")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)# original fualt = 20
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--depth_multiscale",
                                 help="if set, comparing depth maps on multiscales",
                                 action="store_true")
        self.parser.add_argument("--respective_depth_constraint",
                                 help="if set, using respective relationship cross frames",
                                 action="store_true")
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_BCELoss",
                                 help="if set, doesn't apply BCELoss on distance_mask",
                                 action="store_true")
        self.parser.add_argument("--distance_mask_weight",
                                 type=float,
                                 help="distance_mask_BCELoss_weight",
                                 default= 0.2)
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--distance_constraint_mask",
                                 help="if set, uses a generated mask to constrant distance imformation frames",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])
                                 #default=["encoder", "depth", "pose_encoder", "pose","distance_constraint_encoder","distance_constraint_mask"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
