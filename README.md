# Robust Tracking against Adversarial Attacks

:herb: **[Robust Tracking against Adversarial Attacks](https://arxiv.org/pdf/2007.09919.pdf)**

Shuai Jia, Chao Ma, Yibing Song and Xiaokang Yang

*European Conference on Computer Vision (ECCV), 2020*

## Introduction
<img src="https://github.com/joshuajss/RTAA/blob/master/demo/visualization.png" width='700'/><br/>

Deep convolutional neural networks (CNNs) are vulnerable to adversarial attacks. 
- We propose to generate adversarial examples to deteriorate the performance for visual object tracking. 
- Conversely, we propose to defend deep trackers against adversarial attacks that eliminate their effect to alleviate performance drops caused by the adversarial attack.
- We choose two typical trackers, **DaSiamRPN** and **RT-MDnet**.

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
 
:herb: **All raw results are available.**  [[Google_drive]](https://drive.google.com/drive/folders/1E1XrLghxyZRMQSPzO6oxKXd_htErX6WY?usp=sharing)  [[Baidu_Disk]](https://pan.baidu.com/s/1Iqdn34ZXufyxrpvujQykCg) Code: 5ex9

## Quick Start
:herb: **The code of adversarial attack on DaSiamRPN is released!!**
- You should download the OTB2015 dataset in ```data``` folder.
- Please download the pretrained model in [Code_DaSiamRPN](https://github.com/foolwood/DaSiamRPN).

Test the original performance on OTB2015 dataset, please using the follwing command.
```
cd DaSiamRPN/code
python test_otb.py
```
Test the adversarial attack performance on OTB2015 dataset, please using the follwing command.
```
cd DaSiamRPN/code
python test_otb_attack.py
```
Test the adversarial defense performance on OTB2015 dataset, please using the follwing command.
```
cd DaSiamRPN/code
python test_otb_defense.py
```

```-v``` can be used to visualize the tracking results.

## Demo
<img src="https://github.com/joshuajss/RTAA/blob/master/demo/attack_otb100.gif" width='300'/>   <img src="https://github.com/joshuajss/RTAA/blob/master/demo/defense_otb100.gif" width='300'/><br/>
&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; <img src="https://github.com/joshuajss/RTAA/blob/master/demo/legend.png" width='400'/><br/>




## Citation
If any part of our paper and code is helpful to your work, please generously citing: 
```
@inproceedings{jia-eccv20-RTAA,
  title={Robust Tracking against Adversarial Attacks},
  author={Jia, Shuai and Ma, Chao and Song, Yibing and Yang, Xiaokang},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```
```
@inproceedings{zhu-eccv18-dasiamrpn,
  title={Distractor-aware Siamese Networks for Visual Object Tracking},
  author={Zhu, Zheng and Wang, Qiang and Li, Bo and Wu, Wei and Yan, Junjie and Hu, Weiming},
  booktitle={European Conference on Computer Vision},
  year={2018}
}
```
```
@InProceedings{jung-eccv19-rtmdnet,
author = {Jung, Ilchae and Son, Jeany and Baek, Mooyeol and Han, Bohyung},
title = {Real-Time MDNet},
booktitle = {European Conference on Computer Vision (ECCV)},
month = {Sept},
year = {2018}
}
```

Thank you!

## License
Licensed under an MIT license.
