# 2023-spring-CV-PJ3

## ①代码文件说明：

```
├── ex1
│   ├── best_model     # 存储最佳模型及训练验证数据
│   ├── data     # 存储DUIE数据和处理后的句子表征向量（需自行下载）
├── ex3
│   ├── LLFF     # 用于将相机参数转化为llff格式的代码（需自行下载）
│   ├── ├── ……
│   ├── nerf     # 使用NeRF模型进行重建的代码
│   ├── ├── config     # 存放训练参数设置文件
│   ├── img     # 存放用于重建的图像、相机参数
│   ├── ├── ……
│   ├── nerf.gif     # 重建得到视频转化成的动图
```
LLFF下载链接：https://github.com/Fyusion/LLFF.git

## ②自监督任务训练步骤：
·打开文件夹中的pretrain.py文件，设置训练参数并运行，得到预训练模型

·打开lincls_train.py文件，将其中的mode参数设为'train'，设置其他训练参数

·运行得到训练模型及结果


## ③自监督任务测试步骤
·打开lincls_train.py文件，将其中的mode参数设为'predict'，设置其他训练参数

·设置模型路径，运行即可得到结果

## ④三维重建任务步骤：
·使用COLMAP软件对图像进行参数重建，得到的结果文件放入sparse/0文件夹

·使用LLFF中的img2pose.py文件将参数转化为llff格式

·在nerf.config文件中新建txt文件，输入训练参数、图像地址等信息

·使用python run_nerf.py --config configs/<.txt> --render only进行参数重建

·运行得到视频结果

