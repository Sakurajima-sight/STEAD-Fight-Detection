import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='STEAD')  # 创建一个ArgumentParser对象，描述为STEAD模型的参数
    parser.add_argument('--rgb_list', default='ucf_x3d_train.txt', help='输入的RGB特征列表文件')  # 训练时使用的RGB特征列表文件路径
    parser.add_argument('--test_rgb_list', default='ucf_x3d_test.txt', help='测试时的RGB特征列表文件')  # 测试时使用的RGB特征列表文件路径

    parser.add_argument('--comment', default='tiny', help='训练模型时保存的ckpt名称的备注')  # 用于命名模型的备注（如：tiny）

    parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout的比例')  # dropout的比例，控制神经网络的正则化
    parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='attention层的dropout比例')  # attention层的dropout比例，避免过拟合
    parser.add_argument('--lr', type=str, default=2e-4, help='学习率，默认值为2e-4')  # 学习率，默认设置为2e-4
    parser.add_argument('--batch_size', type=int, default=16, help='每批次的数据样本数量（默认：16）')  # 每批次的样本数，默认是16

    parser.add_argument('--model_name', default='model', help='保存模型时的名称')  # 模型保存时的名称
    parser.add_argument('--pretrained_ckpt', default= None, help='预训练模型的检查点路径')  # 预训练模型的路径（可选）
    parser.add_argument('--max_epoch', type=int, default=30, help='最大训练轮数（默认：30）')  # 最大训练轮数，默认是30
    parser.add_argument('--warmup', type=int, default=1, help='预热轮数')  # 预热训练的轮数，通常用于逐步增加学习率

    args = parser.parse_args()  # 解析命令行参数
    return args  # 返回解析后的参数
