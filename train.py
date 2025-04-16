import torch
import torch.nn.functional as F
import option
args = option.parse_args()
from torch import nn
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve

torch.autograd.set_detect_anomaly(True)  # 开启自动梯度异常检测

# 三元组损失函数（Triplet Loss）
class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    # 计算两个向量之间的欧氏距离
    def distance(self, x, y):
        d = torch.cdist(x, y, p=2)  # p=2表示使用欧氏距离
        return d

    def forward(self, feats, margin = 100.0):
        # feats: 特征向量，margin: 边距（用于控制正负样本间的最小间隔）
        bs = len(feats)  # 获取批次大小
        n_feats = feats[:bs // 2]  # 负样本特征
        a_feats = feats[bs // 2:]  # 正样本特征
        n_d = self.distance(n_feats, n_feats)  # 计算负样本之间的距离
        a_d = self.distance(n_feats, a_feats)  # 计算负样本和正样本之间的距离
        n_d_max, _ = torch.max(n_d, dim=0)  # 取负样本之间的最大距离
        a_d_min, _ = torch.min(a_d, dim=0)  # 取负样本和正样本之间的最小距离
        a_d_min = margin - a_d_min  # 计算最小距离与边距的差值
        a_d_min = torch.max(torch.zeros(bs // 2).cuda(), a_d_min)  # 确保差值不为负数
        return torch.mean(n_d_max) + torch.mean(a_d_min)  # 返回平均最大负样本距离和最小正样本距离

# 综合损失函数（结合了交叉熵损失和三元组损失）
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()  # 二元交叉熵损失（用于二分类）
        self.triplet = TripletLoss()  # 三元组损失

    def forward(self, scores, feats, targets, alpha = 0.01):
        loss_ce = self.criterion(scores, targets)  # 计算交叉熵损失
        loss_triplet = self.triplet(feats)  # 计算三元组损失
        return loss_ce, alpha * loss_triplet  # 返回总损失（交叉熵损失和三元组损失的加权和）

# 训练函数
def train(loader, model, optimizer, scheduler, device, epoch):

    with torch.set_grad_enabled(True):  # 开启梯度计算
        model.train()  # 设置模型为训练模式
        pred = []  # 存储预测结果
        label = []  # 存储真实标签
        for step, (ninput, nlabel, ainput, alabel) in tqdm(enumerate(loader)):  # 遍历训练数据集
            input = torch.cat((ninput, ainput), 0).to(device)  # 合并正负样本输入并送到设备
            
            scores, feats, = model(input)  # 前向传播，获得预测分数和特征向量
            pred += scores.cpu().detach().tolist()  # 将预测结果保存到列表中
            labels = torch.cat((nlabel, alabel), 0).to(device)  # 合并正负样本标签并送到设备
            label += labels.cpu().detach().tolist()  # 将真实标签保存到列表中

            loss_criterion = Loss()  # 初始化损失函数
            loss_ce, loss_con = loss_criterion(scores.squeeze(), feats, labels)  # 计算损失
            loss = loss_ce + loss_con  # 总损失是交叉熵损失和三元组损失的和

            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度

            optimizer.step()  # 更新模型参数
            scheduler.step_update(epoch * len(loader) + step)  # 更新学习率

        # 计算ROC曲线和PR曲线的AUC值
        fpr, tpr, _ = roc_curve(label, pred)  # 计算假阳性率和真阳性率
        roc_auc = auc(fpr, tpr)  # 计算AUC值
        precision, recall, _ = precision_recall_curve(label, pred)  # 计算精确度和召回率
        pr_auc = auc(recall, precision)  # 计算PR-AUC值
        print('train_pr_auc : ' + str(pr_auc))  # 输出PR-AUC值
        print('train_roc_auc : ' + str(roc_auc))  # 输出ROC-AUC值
        return loss.item()  # 返回损失值
