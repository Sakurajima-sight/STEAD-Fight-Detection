from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
import numpy as np
from utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import summary
from tqdm import tqdm
import option
args = option.parse_args()
from model import Model
from dataset import Dataset
from train import train
from test import test
import datetime
import os
import random


# 保存配置信息到文件
def save_config(save_path):
    path = save_path + '/'
    os.makedirs(path, exist_ok=True)  # 如果目录不存在则创建
    f = open(path + "config_{}.txt".format(datetime.datetime.now()), 'w')  # 创建配置文件
    for key in vars(args).keys():
        f.write('{}: {}'.format(key, vars(args)[key]))  # 写入每个参数
        f.write('\n')

# 根据学习率、批次大小和备注创建保存路径
savepath = './ckpt/{}_{}_{}'.format(args.lr, args.batch_size, args.comment)
save_config(savepath)

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):  # 如果是全连接层
        torch.nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀分布初始化权重
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  # 初始化偏置为零
    elif isinstance(m, nn.Conv1d):  # 如果是1D卷积层
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用He初始化
    elif isinstance(m, nn.Conv2d):  # 如果是2D卷积层
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv3d):  # 如果是3D卷积层
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
if __name__ == '__main__':
    args = option.parse_args()  # 解析命令行参数
    random.seed(2025)  # 设置随机种子
    np.random.seed(2025)  # 设置numpy的随机种子
    torch.cuda.manual_seed(2025)  # 设置PyTorch的随机种子
    device = torch.device('cuda')  # 使用GPU设备

    # 数据加载器，训练时不进行数据乱序，乱序由Dataset类处理
    train_loader = DataLoader(Dataset(args, test_mode=False),
                               batch_size=args.batch_size // 2)  # 使用一半的batch大小训练
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=args.batch_size)  # 使用标准的batch大小测试

    # 初始化模型
    model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    model.apply(init_weights)  # 初始化模型的权重

    # 加载预训练模型（如果有的话）
    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)  # 加载预训练模型
        model.load_state_dict(model_ckpt)  # 加载模型参数
        print("pretrained loaded")

    model = model.to(device)  # 将模型移到GPU

    # 如果保存目录不存在，创建该目录
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.2)

    num_steps = len(train_loader)  # 获取训练数据的步数
    # 设置学习率调度器，使用余弦退火学习率策略
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.max_epoch * num_steps,  # 初始的总训练步数
            cycle_mul=1.,
            lr_min=args.lr * 0.2,  # 最小学习率
            warmup_lr_init=args.lr * 0.01,  # 预热阶段的初始学习率
            warmup_t=args.warmup * num_steps,  # 预热阶段的步数
            cycle_limit=20,  # 循环次数
            t_in_epochs=False,
            warmup_prefix=True,  # 启用预热策略
            cycle_decay=0.95,  # 每次周期的衰减系数
        )

    test_info = {"epoch": [], "test_AUC": [], "test_PR": []}  # 用于保存测试信息

    # 开始训练
    for step in tqdm(
            range(0, args.max_epoch),  # 训练的最大轮数
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        # 训练阶段
        cost = train(train_loader, model, optimizer, scheduler, device, step)
        scheduler.step(step + 1)  # 更新学习率

        # 测试阶段
        auc, pr_auc = test(test_loader, model, args, device)

        # 保存每轮的测试结果
        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        test_info["test_PR"].append(pr_auc)
        
        # 保存模型
        torch.save(model.state_dict(), savepath + '/' + args.model_name + '{}-x3d.pkl'.format(step))
        save_best_record(test_info, os.path.join(savepath + "/", '{}-step.txt'.format(step)))

    # 保存最终模型
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
