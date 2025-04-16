import torch
from torch import nn
from utils import FeedForward, FOCUS
from performer_pytorch import Performer

# 注意力模块
class AttnBlock(nn.Module):
    def __init__(self, dim, depth, dropout, attn_dropout, heads = 16, ff_mult = 2):
        super().__init__()
        # 使用 Performer 作为注意力机制
        self.performer = Performer(dim = dim, 
                                   depth = depth, 
                                   heads = heads, 
                                   dim_head = dim // heads, 
                                   causal = False,
                                   ff_mult = ff_mult,
                                   local_attn_heads = 8,
                                   local_window_size = dim // 8,
                                   ff_dropout = dropout,
                                   attn_dropout = attn_dropout,
                                   )

    def forward(self, x):
        B, T, H, W, C = x.shape
        # 将输入数据展开为二维的形式，供 Performer 使用
        x = x.view(B, -1, C)
        x = self.performer(x)  # 通过 Performer 计算注意力
        # 将输出数据恢复为原始形状
        x = x.view(B, T, H, W, C)
        return x

# 卷积模块
class ConvBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        dropout = 0.,
        heads = 16,
    ):
        super().__init__()
        # 第一个归一化层
        self.norm1 = nn.LayerNorm(dim)
        # 第二个归一化层
        self.norm2 = nn.LayerNorm(dim)
        # FOCUS 卷积操作
        self.conv = FOCUS(dim, heads)
        # 前馈神经网络
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        # 通过卷积模块后添加残差连接
        x = x + self.conv(self.norm1(x))
        # 通过前馈网络模块后添加残差连接
        x = x + self.ff(self.norm2(x))
        return x

# 主模型
class Model(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.2,
        attn_dropout = 0.1,
        ff_mult = 1,
    ):
        dims = (32, 32)  # 每一层的特征维度
        depths = (1, 1)  # 每一层的深度
        block_types = ('c', 'a')  # 每一层的模块类型（'c'为卷积，'a'为注意力）
        super().__init__()
        self.init_dim, *_, last_dim = dims  # 获取初始维度和最后一层的维度

        # 存储各个阶段的模块
        self.stages = nn.ModuleList([])

        # 构建各个阶段的网络层
        for ind, (depth, block_types) in enumerate(zip(depths, block_types)):
            is_last = ind == len(depths) - 1  # 判断是否为最后一层
            stage_dim = dims[ind]
            
            if block_types == "c":
                for _ in range(depth):
                    # 添加卷积块
                    self.stages.append(
                        ConvBlock(
                            dim = stage_dim,
                            ff_mult=ff_mult,
                            dropout = dropout,
                        )
                    )
            elif block_types == "a":
                for _ in range(depth):
                    # 添加注意力块
                    self.stages.append(AttnBlock(stage_dim, 1, dropout, attn_dropout, ff_mult=ff_mult))
                
            if not is_last:
                # 非最后一层添加线性层和归一化层
                self.stages.append(
                    nn.Sequential(
                        nn.LayerNorm(stage_dim),
                        nn.Linear(stage_dim, dims[ind + 1])
                    )
                )

        # 初始化归一化层、全连接层和池化层
        self.norm0 = nn.LayerNorm(192)
        self.linear = nn.Linear(192, dims[0])
        self.norm = nn.LayerNorm(last_dim)
        self.fc = nn.Linear(last_dim, 1)
        self.drop_out = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        
    def forward(self, x):
        # 交换维度，使得模型能够处理输入的顺序
        x = x.permute(0, 2, 3, 4, 1)

        # 如果输入维度与初始化维度不一致，则通过线性层处理
        if x.shape[4] != self.init_dim:
            x = self.linear(self.norm0(x))

        # 遍历每一层的模块
        for stage in self.stages:
            x = stage(x)

        # 恢复维度顺序
        x = x.permute(0, 4, 1, 2, 3)
        # 使用自适应池化层将数据大小缩放为(1, 1, 1)
        x = self.pooling(x).squeeze()

        # 使用丢弃层，归一化层和全连接层输出最终的预测结果
        x = self.drop_out(x)
        x = self.norm(x)
        logits = self.fc(x)
        return logits, x
