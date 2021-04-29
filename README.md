# This is a repo discribing our depth estimation method
 Most code based on [monodepth2](https://github.com/nianticlabs/monodepth2).

 Our work is accepted by CVMP2020, [Constant Velocity Constraints for Self-Supervised Monocular Depth Estimation](https://dl.acm.org/doi/pdf/10.1145/3429341.3429355)
# Comparision Results 
*(all models trained in a self-supervised manner and test on a same metrics and testset)*

| Methods    | abs rel | sq rel | RMSE  | RMSE log | D1 |
| :----------- | :-----: | :----: | :---: | :------: | :--------: |
| Monodepth2 (ICCV2019) | 0.115   | 0.882  | 4.701 | 0.190    | 0.879      |
| FeatDepth (ECCV2020) | 0.104 | 0.729 | 4.481 | 0.179 | 0.893|
| PackNet (CVPR2020) | 0.111 | 0.785 | 4.601 | 0.189 | 0.878 |
| Fisheyedepthnet (ICRA2020) | 0.117 | 0.867 | 4.739 | 0.190 | 0.869 |
| **Ours** | 0.112 | 0.816 | 4.715 | 0.190 | 0.880 |
## New comparison:
| Methods |abs rel|sq rel| RMSE |rmse log | D1 | D2 | D3 |
|Hr-depth(AAAI 2021)|0.109|0.792|4.632|0.185|0.884|0.962|0.983|
|hrnet-18|0.109|0.799|4.628|0.186|0.887| 0.963|0.982|
|hrnet-18 + vc|0.107|0.767|4.553|0.185|0.889|0.963|0.982|
|hrnet-32|0.104|0.838|4.638|0.183|0.895|0.963|0.982|


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
```
sh test_sample.sh
```
