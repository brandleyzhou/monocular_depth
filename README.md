# This is a repo discribing our depth estimation method
# Results

| Methods    | abs rel | sq rel | RMSE  | RMSE log | D1 |
| :----------- | :-----: | :----: | :---: | :------: | :--------: |
| Monodepth2(ICCV2019) | 0.115   | 0.882  | 4.701 | 0.190    | 0.879      |
| FeatDepth(ECCV2020) | 0.104 | 0.729 | 4.481 | 0.179 | 0.893|
| PackNet(CVPR2020) | 0.111 | 0.785 | 4.601 | 0.189 | 0.878 |
| **Ours** | 0.112 | 0.816 | 4.715 | 0.190 | 0.880 |

![](demo1.gif)
# whentesting on non-rigid objects
![](pedestrians.gif)
*train.py* just uses options defined in *options.py* to instance a trainer object defined in *trainer.py*
### Model networks in *networks/*

