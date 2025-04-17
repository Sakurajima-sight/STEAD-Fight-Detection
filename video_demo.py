from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, \
    QSlider, QStyle, QSizePolicy, QFileDialog
import sys
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtCore import Qt, QUrl

# 创建一个 QWidget 子类表示应用程序窗口
class Window(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口的属性，如标题、大小和图标
        self.setWindowTitle("PyQt5 媒体播放器")
        self.setGeometry(350, 100, 700, 500)  # 设置窗口的位置和大小
        self.setWindowIcon(QIcon('player.png'))  # 设置窗口图标

        # 设置窗口背景颜色为黑色
        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        # 初始化用户界面
        self.init_ui()

        # 显示窗口
        self.show()

    # 初始化用户界面组件
    def init_ui(self):
        # 创建一个 QMediaPlayer 对象
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # 创建一个 QVideoWidget 用于显示视频
        videowidget = QVideoWidget()

        # 创建一个 QPushButton 用于打开视频文件
        openBtn = QPushButton('打开视频')
        openBtn.clicked.connect(self.open_file)  # 点击按钮时调用 open_file 方法

        # 创建一个 QPushButton 用于播放或暂停视频
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)  # 初始时禁用播放按钮
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  # 设置播放按钮图标
        self.playBtn.clicked.connect(self.play_video)  # 点击时切换播放或暂停

        # 创建一个 QSlider 用于视频进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)  # 初始时视频进度为0
        self.slider.sliderMoved.connect(self.set_position)  # 移动滑块时调用 set_position 方法

        # 创建一个 QLabel 用于显示视频信息或错误信息
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)  # 设置标签的大小策略

        # 创建一个 QHBoxLayout 用于水平布局
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)  # 设置水平布局的外边距为0

        # 向水平布局中添加按钮和进度条
        hboxLayout.addWidget(openBtn)
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)

        # 创建一个 QVBoxLayout 用于垂直布局
        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(videowidget)  # 添加视频显示控件
        vboxLayout.addLayout(hboxLayout)  # 添加按钮和进度条
        vboxLayout.addWidget(self.label)  # 添加标签

        # 设置窗口的布局
        self.setLayout(vboxLayout)

        # 设置视频输出为视频控件
        self.mediaPlayer.setVideoOutput(videowidget)

        # 连接媒体播放器的信号和槽函数
        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)  # 播放状态变化时调用 mediastate_changed 方法
        self.mediaPlayer.positionChanged.connect(self.position_changed)  # 视频位置变化时调用 position_changed 方法
        self.mediaPlayer.durationChanged.connect(self.duration_changed)  # 视频时长变化时调用 duration_changed 方法
        self.mediaPlayer.error.connect(self.handle_errors)  # 错误发生时调用 handle_errors 方法

    # 打开视频文件的方法
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "打开视频")  # 弹出文件选择框

        if filename != '':  # 如果选择了文件
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))  # 设置视频文件
            self.playBtn.setEnabled(True)  # 启用播放按钮

    # 播放或暂停视频的方法
    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:  # 如果视频正在播放
            self.mediaPlayer.pause()  # 暂停视频
        else:
            self.mediaPlayer.play()  # 播放视频

    # 处理媒体播放状态变化（播放或暂停）
    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))  # 设置为暂停图标
        else:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  # 设置为播放图标

    # 处理视频位置变化
    def position_changed(self, position):
        self.slider.setValue(position)  # 更新进度条的值

    # 处理视频时长变化
    def duration_changed(self, duration):
        self.slider.setRange(0, duration)  # 设置进度条的范围为从0到视频的总时长

    # 设置视频播放位置
    def set_position(self, position):
        self.mediaPlayer.setPosition(position)  # 设置媒体播放器的当前位置

    # 处理错误信息
    def handle_errors(self):
        self.playBtn.setEnabled(False)  # 错误发生时禁用播放按钮
        self.label.setText("错误: " + self.mediaPlayer.errorString())  # 显示错误信息

# 创建应用实例
app = QApplication(sys.argv)

# 创建主窗口实例
window = Window()

# 运行应用事件循环
sys.exit(app.exec_())
