#python train.py --respective_depth_constraint --depth_loss_weight 0.5 --add_neighboring_frames 1 --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/raw_kitti
#no2
python train.py --batch_size 12 --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/raw_kitti
#no5
#python train.py  --respective_depth_constraint --depth_loss_weight 0.0001 --add_neighboring_frames 1 --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/raw_kitti --load_weights_folder ~/test_monodepth/plan1_sigm_checkpoints_all_v0_new_flipping/mono_model/models/weights_17 
