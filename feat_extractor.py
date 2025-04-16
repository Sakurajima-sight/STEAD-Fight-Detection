import torch
from torch import nn
import numpy as np
from tqdm import tqdm
model_name = 'x3d_l'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)  # 加载预训练模型
from torchinfo import summary
import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.v2 import CenterCrop, Normalize
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler, UniformClipSampler
from torch.utils.data import DataLoader

# 设置设备为GPU或者CPU
device = "cuda"
model = model.eval()  # 设置模型为评估模式
model = model.to(device)  # 将模型加载到指定设备（GPU）

import os.path

# 定义RGB图像的标准化参数（均值和标准差）
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30  # 每秒帧数
model_transform_params  = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    },
    "x3d_l": {
        "side_size": 320,
        "crop_size": 320,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

# 获取基于模型的转换参数
transform_params = model_transform_params[model_name]

# 定义Permute模块，用于调整张量的维度顺序
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)  # 对输入张量进行维度转换

# 定义转换操作
# 这里的transform是针对slow_R50模型的特定转换
transform =  ApplyTransformToKey(
    key="video",  # 对视频数据进行转换
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),  # 均匀地子采样帧
            Lambda(lambda x: x / 255.0),  # 归一化输入的像素值到[0, 1]之间
            Permute((1, 0, 2, 3)),  # 调整维度顺序
            Normalize(mean, std),  # 标准化数据
            ShortSideScale(size=transform_params["side_size"]),  # 调整较短的边为指定尺寸
            CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),  # 中心裁剪
            Permute((1, 0, 2, 3))  # 再次调整维度顺序
        ]
    ),
)

# 输入片段的持续时间，也取决于所使用的模型
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second

# 删除模型的最后一个块
del model.blocks[-1]

# 输出模型的结构信息
summary(model, (1, 3, 16, 320, 320))

# 读取测试数据列表
test_list = list(open("Anomaly_Detection_splits/Anomaly_Test.txt"))

# 只选择包含"Shooting047"的视频路径
test_list = [path for path in test_list if "Shooting047" in path]

# 根据视频路径和标签创建数据集
test_list = [('Videos/' + path.strip('\n'), {'label': 0 if 'Normal' in path else 1, 'video_label': 'X3D_Videos/' + path.strip('\n')}) 
             for path in test_list if not os.path.isfile('X3D_Videos/' + path.strip('\n')[:-3] + 'npy')]  # 只选择尚未处理的文件

print(len(test_list))  # 打印测试数据的数量

# 创建LabeledVideoDataset对象，包含视频路径和标签
dataset = LabeledVideoDataset(
    labeled_video_paths=test_list,  # 输入数据路径
    clip_sampler=UniformClipSampler(clip_duration),  # 使用均匀采样器来选择视频片段
    transform=transform,  # 应用数据转换
    decode_audio=False  # 不解码音频数据
)

# 创建数据加载器
loader = DataLoader(dataset, batch_size=1)

label = None
current = None

# 遍历数据加载器，处理每个视频片段
for inputs in tqdm(loader):
    preds = model(inputs['video'].to(device)).detach().cpu().numpy()  # 对输入视频进行预测
    for i, pred in enumerate(preds):
        # 如果视频标签变化（表示一个新的视频），保存当前视频的最大预测值
        if inputs['video_label'][i][:-3] != label:
            if label is not None:
                np.save(label + 'npy', current.squeeze())  # 保存当前视频片段的预测结果
            label = inputs['video_label'][i][:-3]  # 更新当前视频标签
            current = pred[None, ...]  # 初始化当前视频的预测结果
        else:
            # 如果视频标签相同，则将当前视频片段的预测值和之前的结果合并并取最大值
            current = np.max(np.concatenate((current, pred[None, ...]), axis=0), axis=0)[None, ...]

# 保存最后一个视频的预测结果
np.save(label + 'npy', current.squeeze())
