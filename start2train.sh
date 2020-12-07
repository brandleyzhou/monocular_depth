#no1
python train.py --apply_distortion False --batch_size 12 --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/train_val 
#no2
#python train.py --perception_weight 1 --apply_distortion False --batch_size 12  --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/raw_kitti
#no3 without distortion
#python train.py --batch_size 12 --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/raw_kitti
