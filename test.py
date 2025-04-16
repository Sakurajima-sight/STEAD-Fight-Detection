from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
args = option.parse_args()
from model import Model
from dataset import Dataset
from torchinfo import summary
import umap
import numpy as np

# 定义类标签
classes = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

MODEL_LOCATION = 'saved_models/'  # 模型存储路径
MODEL_NAME = '888tiny'  # 模型名称
MODEL_EXTENSION = '.pkl'  # 模型扩展名

# 测试函数
def test(dataloader, model, args, device = 'cuda', name = "training", main = False):
    model.to(device)  # 将模型移动到指定设备（GPU/CPU）
    plt.clf()  # 清空当前绘图
    
    with torch.no_grad():  # 在测试阶段不需要计算梯度
        model.eval()  # 将模型设置为评估模式
        pred = []  # 存储预测结果
        labels = []  # 存储真实标签
        feats = []  # 存储特征
        for _, inputs in tqdm(enumerate(dataloader)):  # 遍历数据加载器中的数据
            labels += inputs[1].cpu().detach().tolist()  # 获取标签并存储
            input = inputs[0].to(device)  # 获取输入数据并移动到设备
            scores, feat = model(input)  # 前向传播，得到模型的输出和特征
            scores = torch.nn.Sigmoid()(scores).squeeze()  # 对输出应用Sigmoid激活函数
            pred_ = scores.cpu().detach().tolist()  # 将预测结果移到CPU并转换为列表
            feats += feat.cpu().detach().tolist()  # 存储特征
            pred += pred_  # 存储预测分数

        # 计算ROC曲线和AUC值
        fpr, tpr, threshold = roc_curve(labels, pred)  # 计算假阳性率和真阳性率
        roc_auc = auc(fpr, tpr)  # 计算AUC值
        precision, recall, th = precision_recall_curve(labels, pred)  # 计算精确度和召回率
        pr_auc = auc(recall, precision)  # 计算PR-AUC值
        
        # 打印AUC结果
        print('pr_auc : ' + str(pr_auc))
        print('roc_auc : ' + str(roc_auc))

        # 如果是主训练过程，进行特征可视化
        if main:
            feats = np.array(feats)  # 转换为numpy数组
            fit = umap.UMAP()  # 使用UMAP进行降维
            reduced_feats = fit.fit_transform(feats)  # 降维到2D
            labels = np.array(labels)  # 转换标签为numpy数组
            plt.figure()  # 创建新的图形
            # 绘制UMAP降维后的结果，蓝色表示正常类，红色表示异常类
            plt.scatter(reduced_feats[labels == 0, 0], reduced_feats[labels == 0, 1], c='tab:blue', label='Normal', marker='o')
            plt.scatter(reduced_feats[labels == 1, 0], reduced_feats[labels == 1, 1], c='tab:red', label='Anomaly', marker='*')
            plt.title('UMAP Embedding of Video Features')  # 图形标题
            plt.xlabel('UMAP Dimension 1')  # X轴标签
            plt.ylabel('UMAP Dimension 2')  # Y轴标签
            plt.legend()  # 显示图例
            plt.savefig(name + "_embed.png", bbox_inches='tight')  # 保存图像
            plt.close()  # 关闭图形

        # 返回AUC结果
        return roc_auc, pr_auc


if __name__ == '__main__':
    args = option.parse_args()  # 解析命令行参数
    device = torch.device("cuda")  # 使用GPU设备
    model = Model()  # 初始化模型
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)  # 创建数据加载器

    model = model.to(device)  # 将模型移动到GPU
    summary(model, (1, 192, 16, 10, 10))  # 输出模型的结构信息
    model_dict = model.load_state_dict(torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION))  # 加载预训练模型

    # 调用test函数进行测试并打印AUC值
    auc = test(test_loader, model, args, device, name=MODEL_NAME, main=True)
