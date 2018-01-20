# semseg

---
## semantic segmentation algorithms

这个仓库旨在使用keras实现常用的语义分割算法，主要参考如下：
- [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)
- [Semantic Segmentation using Fully Convolutional Networks over the years](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets)
- [fully convolutional neural network in Keras](https://github.com/keras-team/keras/issues/5369)
- [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras)，和pytorch-semseg差不多，不过模型更少，仅仅只有FCN、SegNet和UNet

---
### 网络实现

- FCN
- RefineNet
- DUC
- DRN
- PSPNet
- ENet，可参考[ent_implements](doc/ent_implements.md)
- ErfNet
- LinkNet
- ...

---
### 数据集实现

- CamVid
- PASCAL VOC
- CityScapes
- ADE20K
- Mapillary Vistas Dataset
- ...

---
### 数据集增加

通过仿射变换来实现数据集增加的方法扩充语义分割数据集。

- [imgaug](https://github.com/aleju/imgaug)
- [Augmentor](https://github.com/mdbloice/Augmentor)
- [joint_transforms.py](https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py) 使用这个脚本实现数据集增加

---
### 依赖

- pytorch
- ...

---
### 数据

- CamVid
- PASCAL VOC
- CityScapes
- ...

---
### 用法

**可视化**

[visdom](https://github.com/facebookresearch/visdom)
[开发相关问题](doc/visdom_problem.md)

```bash
# 在tmux或者另一个终端中开启可视化服务器visdom
python -m visdom.server
# 然后在浏览器中查看127.0.0.1:9097
```

**训练**
```bash
# 训练模型
python train.py
```

**校验**
```bash
# 校验模型
python validate.py
```

**测试**
```bash
# 测试模型
python test.py
```

