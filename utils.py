import numpy as np
import torch
from torch import nn
import option

args = option.parse_args()


def modelsize(model, input, type_size=4):
    # 检查GPU利用率
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('模型 {} : 参数总数: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    # 遍历模型的每一层，计算输出的尺寸
    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    # 计算模型中所有中间变量的总数量
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    # 打印中间变量的数量
    print('模型 {} : 中间变量数量: {:3f} M (不包括反向传播)'.format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('模型 {} : 中间变量数量: {:3f} M (包括反向传播)'.format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def save_best_record(test_info, file_path):
    # 保存最佳记录到文件
    f = open(file_path, "w")
    f.write("epoch: {}\n".format(test_info["epoch"][-1]))
    f.write(str(test_info["test_AUC"][-1]))
    f.write("\n")
    f.write(str(test_info["test_PR"][-1]))
    f.close()

# 前馈网络层
def FeedForward(dim, repe = 4, dropout=0.):
    return nn.Sequential(
        nn.Linear(dim, dim * repe),  # 全连接层
        nn.GELU(),                   # 激活函数
        nn.Dropout(dropout),         # Dropout层
        nn.Linear(dim * repe, dim),  # 全连接层
        nn.GELU(),                   # 激活函数
    )

# 多头关系聚合模块（MHRAs）
class FOCUS(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        kernel = 3
    ):
        super().__init__()
        self.heads = heads
        # self.norm3d = nn.BatchNorm3d(dim)  # 3D BatchNorm，未使用
        self.norm2d = nn.BatchNorm2d(dim)  # 2D BatchNorm
        self.norm1d = nn.BatchNorm1d(dim)  # 1D BatchNorm
        self.conv2d = nn.Conv2d(dim, dim, kernel, padding = kernel // 2, groups = heads)  # 2D卷积层
        self.conv1d = nn.Conv1d(dim, dim, kernel, padding = kernel // 2, groups = heads)  # 1D卷积层
        # self.pool = nn.AdaptiveAvgPool3d(1, 1, 1)  # 3D池化，未使用

    def forward(self, x):
        B, T, H, W, C = x.shape  # B: 批大小, T: 时间步长, H: 高度, W: 宽度, C: 通道数
        # x = x.permute(0, 4, 1, 2, 3)  # 修改维度顺序，未使用
        x = x.view(B * T, C, H, W)  # 重新调整输入的维度，适应卷积
        x = self.norm2d(x)  # 2D BatchNorm
        x = self.conv2d(x)  # 2D卷积操作
        x = x.view(B * H * W, C, T)  # 重新调整维度，适应1D卷积
        x = self.norm1d(x)  # 1D BatchNorm
        x = self.conv1d(x)  # 1D卷积操作
        # x = x.view(B, C, T, H, W)  # 重新调整维度，未使用
        # x = self.norm3d(x)  # 3D BatchNorm，未使用
        x = x.view(B, T, H, W, C)  # 恢复原来的输入维度
        # x = x.permute(0, 2, 3, 4, 1)  # 修改维度顺序，未使用
        return x
