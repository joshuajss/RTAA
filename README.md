# Robust Tracking against Adversarial Attacks

:herb: **Robust Tracking against Adversarial Attacks**

Shuai Jia, Chao Ma, Yibing Song and Xiaokang Yang

*European Conference on Computer Vision (ECCV), 2020*

## Introduction

## Prerequisites 

 The environment follows the tracker you intend to attack：
 
 - *The specific setting and pretrained model for **DaSiamPRN** can refer to [Code_DaSiamRPN](https://github.com/foolwood/DaSiamRPN)*
 
 - *The specific setting and pretrained model for **RT-MDNet** can refer to [Code_RT-MDNet](https://github.com/IlchaeJung/RT-MDNet)*
 
## Results
 #### Result for DaSiamRPN on multiple datasets
|                   | OTB2015<br>OP / DP|  VOT2018<br>A / R / EAO  | VOT2016<br>A / R / EAO | UAV123 <br>OP / DP|
| ------------------| :--------------:  | :----:  |:----: |:----: |
| DaSiamRPN         |  0.658 / 0.886    |   0.585 / 0.272 / 0.380  |0.622 / 0.214 / 0.418|  0.592 / 0.791    |
| DaSiamRPN+RandAtt |  0.586 / 0.799    |  0.571 / 0.529 / 0.223   |0.606 / 0.303 / 0.336|  0.572 / 0.769    |
| DaSiamRPN+Att     |  0.050 / 0.050    |  0.536 / 1.447 / 0.097   |0.521 / 1.631 / 0.078|  0.026 / 0.045    |
| DaSiamRPN+Att+Def |  0.473 / 0.639    |  0.579 / 0.674 / 0.195   |0.581 / 0.722 / 0.211|  0.465 / 0.639    |
| DaSiamRPN+Def     |  0.658 / 0.886    |  0.584 / 0.253 / 0.384   |0.625 / 0.224 / 0.439|  0.592 / 0.792    |
 
 
 
 #### Result for RT-MDNet on multiple datasets
|                   | OTB2015<br>OP / DP|  VOT2018<br>A / R / EAO  | VOT2016<br>A / R / EAO | UAV123 <br>OP / DP|
| ------------------| :--------------:  | :----:  |:----: |:----: |
| RT-MDNet         |  0.643 / 0.876    |   0.533 / 0.567 / 0.176  |0.567 / 0.196 / 0.370|  0.512 / 0.754    |
| RT-MDNet+RandAtt |  0.559 / 0.753    |  0.503 / 0.871 / 0.137   |0.550 / 0.452 / 0.235|  0.491 / 0.728    |
| RT-MDNet+Att     |  0.131 / 0.140    |  0.475 / 1.611 / 0.076   |0.469 / 0.928 / 0.128|  0.079 / 0.128    |
| RT-MDNet+Att+Def |  0.420 / 0.589    |  0.515 / 1.021 / 0.110   |0.531 / 0.494 / 0.225|  0.419 / 0.620    |
| RT-MDNet+Def     |  0.644 / 0.883    |  0.529 / 0.538 / 0.179   |0.540 / 0.168 / 0.364|  0.513 / 0.757    |
 
:snowflake: **All raw results can be downloaded soon！**

## Quick Start
:snowflake: **The code is coming soon!**

## Demo



## Citation
If any part of our paper and code is helpful to your work, please generously citing: 

Thank you!

## License
Licensed under an MIT license.
