import torch.utils.data as data
import numpy as np
import torch
import random
torch.set_float32_matmul_precision('medium')
import option
args=option.parse_args()

# 定义类别列表
classes = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

# 自定义数据集类
class Dataset(data.Dataset):
    def __init__(self, args, test_mode=False):
        # 根据是否为测试模式选择RGB文件列表
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list

        self.test_mode = test_mode  # 标记是否为测试模式
        self.list = list(open(self.rgb_list_file))  # 读取RGB文件列表
        self.n_len = 800  # 正常视频的数量
        self.a_len = len(self.list) - self.n_len  # 非正常视频的数量

    def __getitem__(self, index):
        # 如果不是测试模式，获取训练数据
        if not self.test_mode:
            # 在第一次调用时，初始化正负样本的索引并进行随机打乱
            if index == 0:
                self.n_ind = list(range(self.a_len, len(self.list)))  # 正常视频的索引
                self.a_ind = list(range(self.a_len))  # 非正常视频的索引
                random.shuffle(self.n_ind)  # 打乱正常视频的索引
                random.shuffle(self.a_ind)  # 打乱非正常视频的索引

            # 从正常视频和非正常视频中各选取一个样本
            nindex = self.n_ind.pop()  # 从正常视频中选择一个样本
            aindex = self.a_ind.pop()  # 从非正常视频中选择一个样本

            # 读取正常视频样本
            path = self.list[nindex].strip('\n')
            nfeatures = np.load(path, allow_pickle=True)  # 加载视频特征
            nfeatures = np.array(nfeatures, dtype=np.float32)  # 转换为float32类型
            nlabel = 0.0 if "Normal" in path else 1.0  # 根据文件名判断标签，"Normal"为0，其他为1

            # 读取非正常视频样本
            path = self.list[aindex].strip('\n')
            afeatures = np.load(path, allow_pickle=True)  # 加载视频特征
            afeatures = np.array(afeatures, dtype=np.float32)  # 转换为float32类型
            alabel = 0.0 if "Normal" in path else 1.0  # 根据文件名判断标签，"Normal"为0，其他为1

            return nfeatures, nlabel, afeatures, alabel  # 返回正常视频和非正常视频的特征和标签

        else:
            # 测试模式下，只返回一个样本
            path = self.list[index].strip('\n')
            features = np.load(path, allow_pickle=True)  # 加载视频特征
            label = 0.0 if "Normal" in path else 1.0  # 根据文件名判断标签，"Normal"为0，其他为1
            return features, label  # 返回特征和标签

    def __len__(self):
        # 返回数据集的长度，测试模式返回所有样本，训练模式返回较小的长度
        if self.test_mode:
            return len(self.list)
        else:
            return min(self.a_len, self.n_len)  # 训练时返回正常和非正常视频数量中的较小值
