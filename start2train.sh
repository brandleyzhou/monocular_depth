#no1
#python train.py --batch_size 12 --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/raw_kitti
#no2
python train.py --batch_size 12 --add_neighboring_frames 1 --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/raw_kitti
#no3 without distortion
#python train.py --batch_size 12 --using_v 0 --data_size all --acti_func sigm --plan plan1 --png --data_path data_path/raw_kitti
