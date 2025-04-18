import argparse
import logging
import sys
from os import path
from typing import List

import numpy as np
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl  # pylint: disable=no-name-in-module
from PyQt5.QtGui import QIcon, QPalette  # pylint: disable=no-name-in-module
from PyQt5.QtMultimedia import (  # pylint: disable=no-name-in-module
    QMediaContent,
    QMediaPlayer,
)
from PyQt5.QtMultimediaWidgets import QVideoWidget  # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import QApplication  # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QWidget,
)
from torch import Tensor
from tqdm import tqdm

from torch import nn
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.v2 import CenterCrop, Normalize
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from model import Model  # 请确保您的 STEAD 模型在这个路径下
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2

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
    def __init__(self, video_path, x3d_model_name='x3d_l', device="cuda"):
        # 初始化参数
        self.video_path = video_path
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

    def load_video_data(self):
        video = EncodedVideo.from_path(self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = video.duration  # 获取视频总时长（单位：秒）
        video_data = video.get_clip(start_sec=0, end_sec=total_duration)  # 提取整个视频
        return video_data, fps

    def preprocess_video(self, video_data):
        processed_video = self.transform({"video": video_data})["video"]
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
        video_data = self.load_video_data()

        # 预处理视频数据
        processed_video = self.preprocess_video(video_data["video"])

        # 提取特征
        features = self.extract_features(processed_video)

        # 计算异常分数
        anomaly_score = self.calculate_anomaly_score(features)

        return anomaly_score

def real_time_inference(video_path, model_class, x3d_model_name='x3d_l', device="cuda", sequence_length=None):
    """
    实时推理视频，允许用户设置每次推理的帧序列长度（sequence_length）。

    :param video_path: 视频文件路径
    :param model_class: 模型类，用于初始化 VideoAnomalyDetection
    :param x3d_model_name: 使用的X3D模型类型
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :param sequence_length: 每次推理使用的帧数（如果不指定就用原始视频帧数）
    :return: 异常得分和时间戳
    """
    # 创建 VideoAnomalyDetection 实例
    model = model_class(video_path, x3d_model_name, device)

    # 获取视频剪辑
    video_data, fps = model.load_video_data()
    fps = int(fps)

    if sequence_length is None:
        sequence_length = fps

    # 记录异常得分的列表
    anomaly_scores = []
    timestamps = []

    # 计算视频总时长（秒）
    total_frames = video_data["video"].shape[1]

    # 每次推理使用的帧数是 sequence_length
    for start_frame in range(0, total_frames, sequence_length):
        # 计算结束帧索引
        end_frame = start_frame + sequence_length
        if end_frame > total_frames:
            end_frame = total_frames  # 如果结束帧超出总帧数，则用剩余的所有帧

        # 计算当前时间戳
        sec = start_frame / fps

        # 判断当前帧段是否少于16帧，如果是则丢弃并终止循环
        if end_frame - start_frame < 16:
            break  # 直接跳出整个循环

        # 预处理视频数据
        processed_video = model.preprocess_video(video_data["video"][:, start_frame:end_frame, :, :])

        # 提取特征并计算异常得分
        features = model.extract_features(processed_video)
        anomaly_score = model.calculate_anomaly_score(features)

        # 存储得分和时间戳
        anomaly_scores.append(anomaly_score)
        timestamps.append(sec)

    return anomaly_scores, timestamps


APP_NAME = "Anomaly Media Player"


class MplCanvas(FigureCanvasQTAgg):
    # pylint: disable=unused-argument
    def __init__(self, parent=None, width=5, height=4, dpi=100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class Window(QWidget):
    """基于媒体播放器代码的异常检测GUI

    来源: https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
    """

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon("player.png"))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        self._y_pred = torch.tensor([])  # 存储异常检测结果
        self.duration = None

        self.show()

    def init_ui(self):
        # 创建媒体播放器对象
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # 创建视频播放控件
        videowidget = QVideoWidget()

        # 创建打开视频按钮
        openBtn = QPushButton("打开视频")
        openBtn.clicked.connect(self.open_file)

        # 创建播放按钮
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)

        # 创建滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        # 创建标签
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # 创建网格布局
        gridLayout = QGridLayout()

        # 异常检测信号图
        self.graphWidget = MplCanvas(self, width=5, height=1, dpi=100)

        # 特征提取进度条
        self.pbar = QProgressBar()
        self.pbar.setTextVisible(True)

        # 将控件添加到布局中
        gridLayout.addWidget(self.graphWidget, 0, 0, 1, 5)
        gridLayout.addWidget(videowidget, 1, 0, 5, 5)
        gridLayout.addWidget(openBtn, 6, 0, 1, 1)
        gridLayout.addWidget(self.playBtn, 6, 1, 1, 1)
        gridLayout.addWidget(self.slider, 6, 2, 1, 3)
        gridLayout.addWidget(self.pbar, 7, 0, 1, 5)
        gridLayout.addWidget(self.label, 7, 2, 1, 1)

        self.setLayout(gridLayout)

        self.mediaPlayer.setVideoOutput(videowidget)

        # 连接媒体播放器信号
        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)

        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.positionChanged.connect(self.plot)

        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def open_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "打开视频")

        if filename == "":
            return

        feature_load_message_box = QMessageBox()
        feature_load_message_box.setIcon(QMessageBox.Question)
        feature_load_message_box.setText(
            "从所选视频文件中提取异常分数！"
        )
        feature_load_message_box.addButton(
            "提取异常分数", feature_load_message_box.ActionRole
        )
        feature_load_message_box.buttonClicked.connect(self._features_msgbtn)
        feature_load_message_box.exec_()

        if not path.exists(filename):
            raise FileNotFoundError("所选文件不存在。")

        if self.feature_source == "提取异常分数":
            self.label.setText("正在提取异常分数...")
            video_path = filename
            anomaly_scores, timestamps = real_time_inference(video_path, VideoAnomalyDetection, x3d_model_name='x3d_l', device="cuda", sequence_length=32)

        self._y_pred = np.array(anomaly_scores, dtype=np.float32)

        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
        self.playBtn.setEnabled(True)
        self.label.setText("完成！点击播放按钮")
        

    def _features_msgbtn(self, i) -> None:
        self.feature_source = i.text()  # pylint: disable=attribute-defined-outside-init

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    # pylint: disable=unused-argument
    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        else:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.duration = duration

        if self._y_pred is not None:
            self._y_pred = np.repeat(self._y_pred, duration // len(self._y_pred))

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("错误: " + self.mediaPlayer.errorString())

    def plot(self, position):
        if self._y_pred is not None:
            ax = self.graphWidget.axes
            ax.clear()
            ax.set_xlim(0, self.mediaPlayer.duration())
            ax.set_ylim(-0.1, 1.1)
            ax.plot(self._y_pred[:position], "*-", linewidth=7)
            self.graphWidget.draw()


if __name__ == "__main__":
    # 创建应用实例
    app = QApplication(sys.argv)

    # 创建主窗口实例
    window = Window()

    # 运行应用事件循环
    sys.exit(app.exec_())

