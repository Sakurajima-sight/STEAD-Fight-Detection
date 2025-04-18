import torch
from torch import nn
import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo
from torchinfo import summary
from torchvision.transforms.v2 import CenterCrop, Normalize
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from model import Model  # 请确保您的 STEAD 模型在这个路径下
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


# 定义视频转换参数
model_transform_params = {
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

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return torch.permute(x, self.dims)

class VideoAnomalyDetection:
    def __init__(self, video_path, x3d_model_name='x3d_l', frames_per_second=25, device="cuda"):
        # 初始化参数
        self.video_path = video_path
        self.frames_per_second = frames_per_second
        self.device = device
        
        # 定义模型
        self.x3d_model = torch.hub.load('facebookresearch/pytorchvideo', x3d_model_name, pretrained=True)
        self.x3d_model.eval().to(self.device)
        del self.x3d_model.blocks[-1]
        
        # 定义STEAD模型
        self.STEAD_model = Model().to(self.device)
        self.STEAD_model.load_state_dict(torch.load('./ckpt/modelfinal.pkl'))
        self.STEAD_model.eval().to(self.device)
        

        # 获取对应模型的变换参数
        self.transform_params = model_transform_params[x3d_model_name]
        
        # 定义RGB图像的标准化参数（均值和标准差）
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        
        # 定义视频预处理变换
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.transform_params["num_frames"]),
                    Lambda(lambda x: x / 255.0),
                    Permute((1, 0, 2, 3)),
                    Normalize(self.mean, self.std),
                    ShortSideScale(size=self.transform_params["side_size"]),
                    CenterCrop((self.transform_params["crop_size"], self.transform_params["crop_size"])),
                    Permute((1, 0, 2, 3))
                ]
            ),
        )

        # 计算视频剪辑的时长
        self.clip_duration = (self.transform_params["num_frames"] * self.transform_params["sampling_rate"]) / self.frames_per_second

    def get_video_clip(self):
        video = EncodedVideo.from_path(self.video_path)
        total_duration = video.duration  # 获取视频总时长（单位：秒）
        video_data = video.get_clip(start_sec=0, end_sec=total_duration)  # 提取整个视频
        return video_data

    def preprocess_video(self, video_data):
        processed_video = self.transform({"video": video_data["video"]})["video"]
        return processed_video.unsqueeze(0).to(self.device)  # 扩展维度并移动到设备

    def extract_features(self, processed_video):
        with torch.no_grad():
            pred = self.x3d_model(processed_video)  # 获取模型预测
            pred = pred.cpu().numpy()  # 将结果从GPU转回CPU
        return pred

    def calculate_anomaly_score(self, features):
        frame_features = torch.tensor(features.squeeze(0), dtype=torch.float32).unsqueeze(0).to(self.device)
        scores, _ = self.STEAD_model(frame_features)
        anomaly_score = torch.nn.Sigmoid()(scores).item()
        return anomaly_score

    def get_anomaly_score(self):
        # 获取视频剪辑
        video_data = self.get_video_clip()

        # 预处理视频数据
        processed_video = self.preprocess_video(video_data)

        # 提取特征
        features = self.extract_features(processed_video)

        # 计算异常分数
        anomaly_score = self.calculate_anomaly_score(features)

        return anomaly_score


def calculate_accuracy(fight_folder, noFight_folder, model, error_log_file='errors.txt'):
    # 统计总数和正确预测数
    total = 0
    correct = 0
    errors = []  # 用于记录预测错误的路径

    # 遍历fight文件夹中的视频
    for video_file in os.listdir(fight_folder):
        video_path = os.path.join(fight_folder, video_file)
        if video_path.endswith('.mp4'):  # 确保是MP4文件
            video_anomaly_detector = model(video_path)
            anomaly_score = video_anomaly_detector.get_anomaly_score()
            
            # 根据异常分数判断是否为fight
            predicted_label = 'fight' if anomaly_score > 0.9 else 'noFight'
            actual_label = 'fight'
            
            # 统计正确预测
            total += 1
            if predicted_label == actual_label:
                correct += 1
            else:
                errors.append(video_path)  # 记录错误的路径
    
    # 遍历noFight文件夹中的视频
    for video_file in os.listdir(noFight_folder):
        video_path = os.path.join(noFight_folder, video_file)
        if video_path.endswith('.mp4'):  # 确保是MP4文件
            video_anomaly_detector = model(video_path)
            anomaly_score = video_anomaly_detector.get_anomaly_score()
            
            # 根据异常分数判断是否为fight
            predicted_label = 'fight' if anomaly_score > 0.9 else 'noFight'
            actual_label = 'noFight'
            
            # 统计正确预测
            total += 1
            if predicted_label == actual_label:
                correct += 1
            else:
                errors.append(video_path)  # 记录错误的路径
    
    # 将预测错误的路径写入txt文件
    with open(error_log_file, 'w') as f:
        for error_path in errors:
            f.write(f"{error_path}\n")
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    return accuracy

# 设置文件夹路径
fight_folder = 'fight'  # 替换为实际路径
noFight_folder = 'noFight'  # 替换为实际路径

# 创建VideoAnomalyDetection模型实例并传递给calculate_accuracy
video_anomaly_detector = VideoAnomalyDetection  # 这里传递模型类

accuracy = calculate_accuracy(fight_folder, noFight_folder, video_anomaly_detector)

print(f"Accuracy: {accuracy * 100:.2f}%")
