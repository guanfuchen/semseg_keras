# develop相关
下面是开发相关遇到的问题和解决的思路
- 模型的加载和存储，参考[How to Check-Point Deep Learning Models in Keras](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)
- 每一次训练相关的日志应该保存在expermeriments/$DATASET/$MODEL/[logs|weights]，防止不同的训练覆盖相应的日志和权重文件，具体结构可参考[enet-keras](https://github.com/PavlosMelissinos/enet-keras)