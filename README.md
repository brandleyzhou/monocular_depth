# This is a repo discribing our depth estimation method
 Most code based on [monodepth2](https://github.com/nianticlabs/monodepth2)
# Comparision Results

| Methods    | abs rel | sq rel | RMSE  | RMSE log | D1 |
| :----------- | :-----: | :----: | :---: | :------: | :--------: |
| Monodepth2 (ICCV2019) | 0.115   | 0.882  | 4.701 | 0.190    | 0.879      |
| FeatDepth (ECCV2020) | 0.104 | 0.729 | 4.481 | 0.179 | 0.893|
| PackNet (CVPR2020) | 0.111 | 0.785 | 4.601 | 0.189 | 0.878 |
| Fisheyedepthnet (ICRA2020) | 0.117 | 0.867 | 4.739 | 0.190 | 0.869 |
| **Ours** | 0.112 | 0.816 | 4.715 | 0.190 | 0.880 |

# Demo
![](demo1.gif)
## whentesting on non-rigid objects
![](pedestrians.gif)
## API
1. Train models

```
sh start2train.sh
```

2. Evaluating depth
```
sh disp_evaluation.sh
```

3. Testing a sample

'''
sh test_sample.sh
'''
