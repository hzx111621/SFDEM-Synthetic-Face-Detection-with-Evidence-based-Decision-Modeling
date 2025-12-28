import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, average_precision_score
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from PIL import Image
from tqdm import tqdm
import os
import json
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============== 数据增强 ==============
def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform_train, transform_test


# ============== 频域转换 ==============
class FrequencyTransform:
    """将图像转换到频域"""

    @staticmethod
    def img_to_frequency(img_tensor):
        """
        输入: (B, 3, 224, 224) RGB图像
        输出: (B, 6, 224, 224) 频域图像 (3通道幅度 + 3通道相位)
        """
        fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        magnitude_log = torch.log(magnitude + 1e-8)
        freq_img = torch.cat([magnitude_log, phase], dim=1)
        return freq_img


class FrequencyDatasetWrapper:
    """频域数据集包装器"""

    def __init__(self, dataset, convert_to_frequency=True):
        self.dataset = dataset
        self.convert_to_frequency = convert_to_frequency
        self.freq_transform = FrequencyTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.convert_to_frequency:
            img = img.unsqueeze(0)
            freq_img = self.freq_transform.img_to_frequency(img)
            img = freq_img.squeeze(0)

        return img, label


# ============== 步骤1:Real vs Fake 数据集 ==============
class RealFakeDataset(Dataset):
    def __init__(self, gan_path, dm_path, real_path, transform=None, max_samples=None):
        self.data = []
        self.labels = []
        self.transform = transform

        gan_files = list(pathlib.Path(gan_path).glob('**/*.jpg')) + \
                    list(pathlib.Path(gan_path).glob('**/*.png'))
        for img_path in gan_files[:max_samples] if max_samples else gan_files:
            self.data.append(str(img_path))
            self.labels.append(0)

        dm_files = list(pathlib.Path(dm_path).glob('**/*.jpg')) + \
                   list(pathlib.Path(dm_path).glob('**/*.png'))
        for img_path in dm_files[:max_samples] if max_samples else dm_files:
            self.data.append(str(img_path))
            self.labels.append(0)

        real_files = list(pathlib.Path(real_path).glob('**/*.jpg')) + \
                     list(pathlib.Path(real_path).glob('**/*.png'))
        for img_path in real_files[:max_samples * 2] if max_samples else real_files:
            self.data.append(str(img_path))
            self.labels.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return torch.zeros(3, 224, 224), label


# ============== 步骤2:GAN vs Diffusion 数据集 ==============
class GanDmDataset(Dataset):
    def __init__(self, gan_path, dm_path, transform=None, max_samples=None):
        self.data = []
        self.labels = []
        self.transform = transform

        gan_files = list(pathlib.Path(gan_path).glob('**/*.jpg')) + \
                    list(pathlib.Path(gan_path).glob('**/*.png'))
        for img_path in gan_files[:max_samples] if max_samples else gan_files:
            self.data.append(str(img_path))
            self.labels.append(0)

        dm_files = list(pathlib.Path(dm_path).glob('**/*.jpg')) + \
                   list(pathlib.Path(dm_path).glob('**/*.png'))
        for img_path in dm_files[:max_samples] if max_samples else dm_files:
            self.data.append(str(img_path))
            self.labels.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return torch.zeros(3, 224, 224), label


# ============== EDL损失函数 ==============
class EDLLoss(nn.Module):
    def __init__(self, num_classes=2, weights=None):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        if weights is None:
            self.weights = torch.ones(num_classes)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, alpha, labels):
        S = torch.sum(alpha, dim=1)
        one_hot = torch.zeros_like(alpha)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        psi_sum = torch.digamma(S.unsqueeze(1))
        psi_alpha = torch.digamma(alpha)

        weights_batch = self.weights.to(alpha.device)[labels]
        loss = torch.sum(weights_batch.unsqueeze(1) * one_hot * (psi_sum - psi_alpha), dim=1)

        return torch.mean(loss)


# ============== 对比学习损失函数 ==============
class SupConLoss(nn.Module):
    """
    有监督对比学习损失 (Supervised Contrastive Loss)
    论文: https://arxiv.org/abs/2004.11362

    核心思想:
    - 同类样本在特征空间中应该聚集
    - 不同类样本在特征空间中应该分散
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: (batch_size, feature_dim) 归一化后的特征
            labels: (batch_size,) 类标签
            mask: (batch_size, batch_size) 对比对的mask
        """
        assert features.shape[0] == labels.shape[0], \
            f'features和labels的batch size必须相同: {features.shape[0]} vs {labels.shape[0]}'

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]

        if len(features.shape) < 2:
            raise ValueError('`features` 需要至少2个维度')

        # 特征L2归一化
        features = torch.nn.functional.normalize(features, dim=1)

        if labels is not None and mask is not None:
            raise ValueError('不能同时指定 `labels` 和 `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('labels的batch size与features不匹配')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 计算相似度矩阵
        contrast_feature = features
        anchor_feature = features

        # 余弦相似度,增加稳定性
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # 去掉对角线(自己与自己)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out自对比项
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob,加入clip防止inf
        exp_logits = torch.exp(logits) * logits_mask
        log_sum_exp = torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        log_prob = logits - log_sum_exp

        # 计算平均log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        # 防止NaN
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss


# ============== 改进的FADEL模型 - 支持对比学习和可切换激活函数 ==============
class FADELContrastiveModel(nn.Module):
    def __init__(self, backbone='vit_b_16', dropout_rate=0.5, input_channels=3,
                 feature_dim=512, activation='softplus'):
        """
        Args:
            backbone: 骨干网络类型
            dropout_rate: Dropout比率
            input_channels: 输入通道数
            feature_dim: 特征维度
            activation: 证据激活函数类型 ('softplus', 'relu', 'exp')
        """
        super(FADELContrastiveModel, self).__init__()
        self.num_classes = 2
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.backbone_type = backbone
        self.activation_type = activation

        # ViT-B/16 骨干
        if backbone == 'vit_b_16':
            vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            in_features = vit_model.heads.head.in_features

            # 处理输入通道
            if input_channels != 3:
                original_conv = vit_model.conv_proj
                new_conv = nn.Conv2d(input_channels, original_conv.out_channels,
                                     kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride,
                                     padding=original_conv.padding,
                                     bias=False)
                if input_channels == 6:
                    # 对于6通道输入,复制权重并平均
                    new_conv.weight.data = original_conv.weight.data.repeat(1, 2, 1, 1) / 2
                vit_model.conv_proj = new_conv

            # 保存整个模型作为特征提取器(不包括最后的分类头)
            self.vit_model = vit_model
            # 移除原始分类头
            self.vit_model.heads = nn.Identity()
            self.backbone = self.vit_model

        elif backbone == 'efficientnet_b3':
            efficient_net = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            in_features = efficient_net.classifier[1].in_features

            # 处理输入通道
            if input_channels != 3:
                original_conv = efficient_net.features[0][0]
                new_conv = nn.Conv2d(input_channels, original_conv.out_channels,
                                     kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride,
                                     padding=original_conv.padding,
                                     bias=False)
                if input_channels == 6:
                    new_conv.weight.data = original_conv.weight.data.repeat(1, 2, 1, 1) / 2
                efficient_net.features[0][0] = new_conv

            self.backbone = efficient_net.features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            efficient_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = efficient_net.classifier[1].in_features

            if input_channels != 3:
                original_conv = efficient_net.features[0][0]
                new_conv = nn.Conv2d(input_channels, original_conv.out_channels,
                                     kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride,
                                     padding=original_conv.padding,
                                     bias=False)
                if input_channels == 6:
                    new_conv.weight.data = original_conv.weight.data.repeat(1, 2, 1, 1) / 2
                efficient_net.features[0][0] = new_conv

            self.backbone = efficient_net.features
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 【关键】特征投影层 - 用于对比学习
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, feature_dim)
        )

        # EDL证据层 - 用于分类
        self.evidence_layer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, self.num_classes)
        )

        # 【新增】根据配置选择激活函数
        if activation == 'softplus':
            self.activation_fn = nn.Softplus()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'exp':
            self.activation_fn = lambda x: torch.exp(torch.clamp(x, max=10))  # 限制exp输入防止溢出
        else:
            raise ValueError(f"不支持的激活函数类型: {activation}. 请选择 'softplus', 'relu', 或 'exp'")

    def forward(self, x, return_features=False):
        # 骨干网络
        if self.backbone_type == 'vit_b_16':
            # ViT直接返回特征
            x = self.backbone(x)
        else:
            # EfficientNet需要池化
            x = self.backbone(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        # 特征投影
        features = self.feature_proj(x)

        # L2归一化(重要!用于对比学习)
        features_normalized = torch.nn.functional.normalize(features, dim=1)

        # EDL证据
        evidence = self.evidence_layer(features)

        # 【修改】使用配置的激活函数
        evidence = self.activation_fn(evidence)

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        uncertainty = self.num_classes / S

        if return_features:
            return alpha, probs, uncertainty, features_normalized

        return alpha, probs, uncertainty


# ============== 训练函数 ==============
def train_epoch_contrastive(model, dataloader, optimizer, criterion_edl, criterion_contra,
                            device, lambda_contra=0):
    """
    结合EDL损失和对比学习损失的训练
    """
    model.train()
    total_loss = 0.0
    total_edl_loss = 0.0
    total_contra_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播 - 获取特征
        alpha, probs, uncertainty, features = model(imgs, return_features=True)

        # 【损失1】EDL分类损失
        edl_loss = criterion_edl(alpha, labels)

        # 【损失2】对比学习损失
        contra_loss = criterion_contra(features, labels)

        # 【组合】加权损失 - 初期降低对比损失权重
        loss = (1 - lambda_contra) * edl_loss + lambda_contra * contra_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_edl_loss += edl_loss.item()
        total_contra_loss += contra_loss.item()

        predicted = torch.argmax(probs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        pbar.set_postfix({
            'loss': loss.item(),
            'edl': edl_loss.item(),
            'contra': contra_loss.item(),
            'acc': correct / total
        })

    return (total_loss / len(dataloader), correct / total,
            total_edl_loss / len(dataloader), total_contra_loss / len(dataloader),
            np.array(all_labels), np.array(all_preds))


@torch.no_grad()
def evaluate_contrastive(model, dataloader, criterion_edl, criterion_contra, device):
    """评估函数 - 返回特征用于分析"""
    model.eval()
    total_loss = 0.0
    total_edl_loss = 0.0
    total_contra_loss = 0.0
    correct = 0
    total = 0
    all_features = []
    all_probs = []
    all_uncertainty = []
    all_labels = []
    all_preds = []

    for imgs, labels in tqdm(dataloader, desc='Evaluating'):
        imgs, labels = imgs.to(device), labels.to(device)

        alpha, probs, uncertainty, features = model(imgs, return_features=True)

        edl_loss = criterion_edl(alpha, labels)
        contra_loss = criterion_contra(features, labels)
        loss = 0.5 * edl_loss + 0.5 * contra_loss

        total_loss += loss.item()
        total_edl_loss += edl_loss.item()
        total_contra_loss += contra_loss.item()

        predicted = torch.argmax(probs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_features.append(features.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_uncertainty.append(uncertainty.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_preds.append(predicted.cpu().numpy())

    return {
        'loss': total_loss / len(dataloader),
        'edl_loss': total_edl_loss / len(dataloader),
        'contra_loss': total_contra_loss / len(dataloader),
        'acc': correct / total,
        'features': np.concatenate(all_features),
        'probs': np.concatenate(all_probs),
        'uncertainty': np.concatenate(all_uncertainty),
        'labels': np.concatenate(all_labels),
        'preds': np.concatenate(all_preds)
    }


# ============== 特征分离度分析 ==============
def analyze_separation(features, labels, dataset_name=''):
    fake_features = features[labels == 0]
    real_features = features[labels == 1]

    if len(fake_features) > 1:
        fake_distances = pdist(fake_features[:min(200, len(fake_features))], metric='euclidean')
        mean_fake_distance = np.mean(fake_distances)
    else:
        mean_fake_distance = 0

    if len(real_features) > 1:
        real_distances = pdist(real_features[:min(200, len(real_features))], metric='euclidean')
        mean_real_distance = np.mean(real_distances)
    else:
        mean_real_distance = 0

    mean_within_distance = (mean_fake_distance + mean_real_distance) / 2

    fake_center = np.mean(fake_features, axis=0)
    real_center = np.mean(real_features, axis=0)
    center_distance = np.linalg.norm(fake_center - real_center)

    separation_ratio = center_distance / mean_within_distance if mean_within_distance > 0 else 0

    print(f"\n【{dataset_name}特征分离度分析】")
    print(f"  类内距离: {mean_within_distance:.4f}")
    print(f"  类中心距: {center_distance:.4f}")
    print(f"  分离比: {separation_ratio:.4f}", end='')

    return separation_ratio


# ============== 计算AP精度 ==============
def calculate_ap_metrics(labels, probs):
    ap = average_precision_score(labels, probs[:, 1])
    return ap


# ============== Expected Calibration Error (ECE) ==============
def calculate_ece(labels, probs, n_bins=10):
    """
    计算 Expected Calibration Error (预期校准误差)

    Args:
        labels: 真实标签 (numpy array)
        probs: 预测概率 (numpy array, shape: [N, num_classes])
        n_bins: 分箱数量

    Returns:
        ece: Expected Calibration Error
        bin_accuracies: 每个箱的准确率
        bin_confidences: 每个箱的平均置信度
        bin_counts: 每个箱的样本数
    """
    # 获取预测的最大概率和对应的类别
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    # 创建分箱
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前箱中的样本
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            # ECE计算：|准确率 - 置信度| * 该箱样本比例
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin / len(labels)

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(prop_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)

    return ece, np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


# ============== 自定义分类报告(添加accuracy列和ECE) ==============
def print_custom_classification_report(labels, predictions, probs, class_names):
    """
    打印自定义分类报告,添加accuracy列和ECE
    """
    cm = confusion_matrix(labels, predictions)

    # 计算 ECE
    ece, bin_accs, bin_confs, bin_counts = calculate_ece(labels, probs, n_bins=10)

    results = []
    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # 计算每个类的准确率 (正确预测的样本数 / 总样本数)
        class_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        if i == 1:
            ap = average_precision_score(labels, probs[:, 1])
        else:
            ap = average_precision_score(1 - labels, probs[:, 0])

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        support = (labels == i).sum()

        results.append({
            'class': class_names[i],
            'precision': precision,
            'accuracy': class_accuracy,
            'ap': ap,
            'f1': f1,
            'support': support
        })

    # 添加accuracy列
    print(f"{'':12} {'precision':>10} {'accuracy':>10} {'AP':>10} {'f1-score':>10} {'support':>10}")
    print("-" * 75)

    for result in results:
        print(f"{result['class']:12} {result['precision']:10.4f} "
              f"{result['accuracy']:10.4f} {result['ap']:10.4f} "
              f"{result['f1']:10.4f} {int(result['support']):10d}")

    print("-" * 75)

    overall_accuracy = (predictions == labels).sum() / len(labels)
    macro_precision = np.mean([r['precision'] for r in results])
    macro_accuracy = np.mean([r['accuracy'] for r in results])
    macro_ap = np.mean([r['ap'] for r in results])
    macro_f1 = np.mean([r['f1'] for r in results])

    weighted_precision = np.average([r['precision'] for r in results],
                                    weights=[r['support'] for r in results])
    weighted_accuracy = np.average([r['accuracy'] for r in results],
                                   weights=[r['support'] for r in results])
    weighted_ap = np.average([r['ap'] for r in results],
                             weights=[r['support'] for r in results])
    weighted_f1 = np.average([r['f1'] for r in results],
                             weights=[r['support'] for r in results])

    print(f"{'accuracy':12} {overall_accuracy:10.4f}")
    print(f"{'macro avg':12} {macro_precision:10.4f} {macro_accuracy:10.4f} "
          f"{macro_ap:10.4f} {macro_f1:10.4f} {int(sum(r['support'] for r in results)):10d}")
    print(f"{'weighted avg':12} {weighted_precision:10.4f} {weighted_accuracy:10.4f} "
          f"{weighted_ap:10.4f} {weighted_f1:10.4f} {int(sum(r['support'] for r in results)):10d}")

    print("-" * 75)
    print(f"\n{'ECE (预期校准误差)':30} {ece:10.4f}")
    print(f"  (值越小说明模型的置信度校准越好，理想值为0)")


# ============== 可视化函数 - 分别展示 ==============
def plot_train_test_comparison(train_result, test_result, class_names, title_prefix, save_name):
    # 根据类名设置颜色方案
    # Real-绿色, Fake-红色; GAN-紫色, DM-黄色
    if 'Fake' in class_names and 'Real' in class_names:
        # 步骤1: Real vs Fake
        colors = ['#FF6B6B', '#4CAF50']  # Fake-红色, Real-绿色
    elif 'GAN' in class_names and 'Diffusion' in class_names:
        # 步骤2: GAN vs Diffusion
        colors = ['#9C27B0', '#FFC107']  # GAN-紫色, Diffusion-黄色
    else:
        # 默认配色
        colors = ['#FF6B6B', '#45B7D1']

    # 预先计算所有需要的数据
    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    train_tsne = tsne.fit_transform(train_result['features'])

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    test_tsne = tsne.fit_transform(test_result['features'])

    print("Computing PCA...")
    pca_train = PCA(n_components=2)
    train_pca = pca_train.fit_transform(train_result['features'])

    pca_test = PCA(n_components=2)
    test_pca = pca_test.fit_transform(test_result['features'])

    train_unc = train_result['uncertainty'].flatten()
    test_unc = test_result['uncertainty'].flatten()

    train_cm = confusion_matrix(train_result['labels'], train_result['preds'])
    test_cm = confusion_matrix(test_result['labels'], test_result['preds'])

    # 计算性能指标
    train_tp = train_cm[1, 1]
    train_tn = train_cm[0, 0]
    train_fp = train_cm[0, 1]
    train_fn = train_cm[1, 0]

    train_acc = (train_tp + train_tn) / np.sum(train_cm)
    train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
    train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
    train_ap = calculate_ap_metrics(train_result['labels'], train_result['probs'])
    train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if (
                                                                                                train_precision + train_recall) > 0 else 0

    test_tp = test_cm[1, 1]
    test_tn = test_cm[0, 0]
    test_fp = test_cm[0, 1]
    test_fn = test_cm[1, 0]

    test_acc = (test_tp + test_tn) / np.sum(test_cm)
    test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
    test_recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
    test_ap = calculate_ap_metrics(test_result['labels'], test_result['probs'])
    test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (
                                                                                           test_precision + test_recall) > 0 else 0

    train_fpr, train_tpr, _ = roc_curve(train_result['labels'], train_result['probs'][:, 1])
    train_auc = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, _ = roc_curve(test_result['labels'], test_result['probs'][:, 1])
    test_auc = auc(test_fpr, test_tpr)

    # 计算 ECE
    train_ece, train_bin_accs, train_bin_confs, train_bin_counts = calculate_ece(train_result['labels'],
                                                                                 train_result['probs'])
    test_ece, test_bin_accs, test_bin_confs, test_bin_counts = calculate_ece(test_result['labels'],
                                                                             test_result['probs'])

    # 创建所有图表
    plots = []

    # 1. Train t-SNE by Class
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for class_idx, (name, color) in enumerate(zip(class_names, colors)):
        mask = train_result['labels'] == class_idx
        ax1.scatter(train_tsne[mask, 0], train_tsne[mask, 1],
                    c=color, label=name, alpha=0.6, s=30)
    ax1.set_title(f'{title_prefix}Train Set - t-SNE (By Class)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig1, f'{save_name}_1_train_tsne_class.png'))

    # 2. Test t-SNE by Class
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for class_idx, (name, color) in enumerate(zip(class_names, colors)):
        mask = test_result['labels'] == class_idx
        ax2.scatter(test_tsne[mask, 0], test_tsne[mask, 1],
                    c=color, label=name, alpha=0.6, s=30)
    ax2.set_title(f'{title_prefix}Test Set - t-SNE (By Class)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig2, f'{save_name}_2_test_tsne_class.png'))

    # 3. Train t-SNE by Uncertainty
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    scatter3 = ax3.scatter(train_tsne[:, 0], train_tsne[:, 1],
                           c=train_unc, cmap='viridis', s=30, alpha=0.6)
    ax3.set_title(f'{title_prefix}Train Set - t-SNE (By Uncertainty)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Component 1')
    ax3.set_ylabel('Component 2')
    plt.colorbar(scatter3, ax=ax3, label='Uncertainty')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig3, f'{save_name}_3_train_tsne_uncertainty.png'))

    # 4. Test t-SNE by Uncertainty
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    scatter4 = ax4.scatter(test_tsne[:, 0], test_tsne[:, 1],
                           c=test_unc, cmap='viridis', s=30, alpha=0.6)
    ax4.set_title(f'{title_prefix}Test Set - t-SNE (By Uncertainty)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Component 1')
    ax4.set_ylabel('Component 2')
    plt.colorbar(scatter4, ax=ax4, label='Uncertainty')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig4, f'{save_name}_4_test_tsne_uncertainty.png'))

    # 5. Train PCA
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    for class_idx, (name, color) in enumerate(zip(class_names, colors)):
        mask = train_result['labels'] == class_idx
        ax5.scatter(train_pca[mask, 0], train_pca[mask, 1],
                    c=color, label=name, alpha=0.6, s=30)
    ax5.set_title(f'{title_prefix}Train Set - PCA', fontsize=14, fontweight='bold')
    ax5.set_xlabel('PC 1')
    ax5.set_ylabel('PC 2')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig5, f'{save_name}_5_train_pca.png'))

    # 6. Test PCA
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    for class_idx, (name, color) in enumerate(zip(class_names, colors)):
        mask = test_result['labels'] == class_idx
        ax6.scatter(test_pca[mask, 0], test_pca[mask, 1],
                    c=color, label=name, alpha=0.6, s=30)
    ax6.set_title(f'{title_prefix}Test Set - PCA', fontsize=14, fontweight='bold')
    ax6.set_xlabel('PC 1')
    ax6.set_ylabel('PC 2')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig6, f'{save_name}_6_test_pca.png'))

    # 7. Train Confusion Matrix
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    ax7.imshow(train_cm, interpolation='nearest', cmap='Blues')
    ax7.set_title(f'{title_prefix}Train Set - Confusion Matrix', fontsize=14, fontweight='bold')
    ax7.set_ylabel('True Label')
    ax7.set_xlabel('Predicted Label')
    tick_marks = np.arange(len(class_names))
    ax7.set_xticks(tick_marks)
    ax7.set_yticks(tick_marks)
    ax7.set_xticklabels(class_names)
    ax7.set_yticklabels(class_names)
    for i in range(train_cm.shape[0]):
        for j in range(train_cm.shape[1]):
            ax7.text(j, i, str(train_cm[i, j]), ha='center', va='center',
                     color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plots.append((fig7, f'{save_name}_7_train_cm.png'))

    # 8. Test Confusion Matrix
    fig8, ax8 = plt.subplots(figsize=(8, 6))
    ax8.imshow(test_cm, interpolation='nearest', cmap='Blues')
    ax8.set_title(f'{title_prefix}Test Set - Confusion Matrix', fontsize=14, fontweight='bold')
    ax8.set_ylabel('True Label')
    ax8.set_xlabel('Predicted Label')
    ax8.set_xticks(tick_marks)
    ax8.set_yticks(tick_marks)
    ax8.set_xticklabels(class_names)
    ax8.set_yticklabels(class_names)
    for i in range(test_cm.shape[0]):
        for j in range(test_cm.shape[1]):
            ax8.text(j, i, str(test_cm[i, j]), ha='center', va='center',
                     color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plots.append((fig8, f'{save_name}_8_test_cm.png'))

    # 9. Train ROC Curve
    fig9, ax9 = plt.subplots(figsize=(8, 6))
    ax9.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'AUC = {train_auc:.3f}')
    ax9.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax9.set_xlim([0.0, 1.0])
    ax9.set_ylim([0.0, 1.05])
    ax9.set_xlabel('False Positive Rate')
    ax9.set_ylabel('True Positive Rate')
    ax9.set_title(f'{title_prefix}Train Set - ROC Curve', fontsize=14, fontweight='bold')
    ax9.legend(loc="lower right")
    ax9.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig9, f'{save_name}_9_train_roc.png'))

    # 10. Test ROC Curve
    fig10, ax10 = plt.subplots(figsize=(8, 6))
    ax10.plot(test_fpr, test_tpr, color='darkorange', lw=2, label=f'AUC = {test_auc:.3f}')
    ax10.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax10.set_xlim([0.0, 1.0])
    ax10.set_ylim([0.0, 1.05])
    ax10.set_xlabel('False Positive Rate')
    ax10.set_ylabel('True Positive Rate')
    ax10.set_title(f'{title_prefix}Test Set - ROC Curve', fontsize=14, fontweight='bold')
    ax10.legend(loc="lower right")
    ax10.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig10, f'{save_name}_10_test_roc.png'))

    # 11. Performance Metrics Comparison
    fig11, ax11 = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'AP', 'F1-Score']
    train_values = [train_acc, train_precision, train_recall, train_ap, train_f1]
    test_values = [test_acc, test_precision, test_recall, test_ap, test_f1]

    x = np.arange(len(metrics))
    width = 0.35

    ax11.bar(x - width / 2, train_values, width, label='Train', color='#45B7D1', alpha=0.7)
    ax11.bar(x + width / 2, test_values, width, label='Test', color='#FF6B6B', alpha=0.7)
    ax11.set_ylabel('Score')
    ax11.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax11.set_xticks(x)
    ax11.set_xticklabels(metrics, rotation=15)
    ax11.set_ylim([0, 1.1])
    ax11.legend()
    ax11.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plots.append((fig11, f'{save_name}_11_metrics.png'))

    # 12. Uncertainty Distribution
    fig12, ax12 = plt.subplots(figsize=(8, 6))
    ax12.hist(train_unc, bins=30, alpha=0.6, label='Train', color='#45B7D1', edgecolor='black')
    ax12.hist(test_unc, bins=30, alpha=0.6, label='Test', color='#FF6B6B', edgecolor='black')
    ax12.set_xlabel('Uncertainty')
    ax12.set_ylabel('Frequency')
    ax12.set_title('Uncertainty Distribution Comparison', fontsize=14, fontweight='bold')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig12, f'{save_name}_12_uncertainty.png'))

    # 13. Train Calibration Curve (ECE)
    fig13, ax13 = plt.subplots(figsize=(8, 6))
    bin_centers = np.linspace(0.05, 0.95, 10)
    ax13.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax13.bar(bin_centers, train_bin_accs, width=0.08, alpha=0.7,
             color='#45B7D1', edgecolor='black', label='Accuracy')
    ax13.plot(bin_centers, train_bin_confs, 'ro-', linewidth=2,
              markersize=8, label='Confidence')
    ax13.set_xlabel('Confidence', fontsize=12)
    ax13.set_ylabel('Accuracy', fontsize=12)
    ax13.set_title(f'{title_prefix}Train Set - Calibration Curve (ECE={train_ece:.4f})',
                   fontsize=14, fontweight='bold')
    ax13.set_xlim([0, 1])
    ax13.set_ylim([0, 1])
    ax13.legend(loc='upper left')
    ax13.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig13, f'{save_name}_13_train_calibration.png'))

    # 14. Test Calibration Curve (ECE)
    fig14, ax14 = plt.subplots(figsize=(8, 6))
    ax14.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax14.bar(bin_centers, test_bin_accs, width=0.08, alpha=0.7,
             color='#FF6B6B', edgecolor='black', label='Accuracy')
    ax14.plot(bin_centers, test_bin_confs, 'ro-', linewidth=2,
              markersize=8, label='Confidence')
    ax14.set_xlabel('Confidence', fontsize=12)
    ax14.set_ylabel('Accuracy', fontsize=12)
    ax14.set_title(f'{title_prefix}Test Set - Calibration Curve (ECE={test_ece:.4f})',
                   fontsize=14, fontweight='bold')
    ax14.set_xlim([0, 1])
    ax14.set_ylim([0, 1])
    ax14.legend(loc='upper left')
    ax14.grid(True, alpha=0.3)
    plt.tight_layout()
    plots.append((fig14, f'{save_name}_14_test_calibration.png'))

    # 逐个显示和保存
    print("\nProcessing and saving plots one by one...")
    for idx, (fig, filename) in enumerate(plots, 1):
        print(f"\n[{idx}/{len(plots)}] Processing: {filename}")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.show()
        plt.close(fig)

    print(f"\n✓ All {len(plots)} plots have been processed and saved!")


# ============== 模型管理器 ==============
class ModelManager:
    def __init__(self, model_dir='models_vit', step_name='step1'):
        safe_step_name = step_name.replace(':', '').replace('/', '_')
        self.model_dir = pathlib.Path(model_dir) / safe_step_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, optimizer, metrics, epoch, model_name='fadel_vit'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.model_dir / f'{model_name}_{timestamp}.pth'
        config_path = self.model_dir / f'{model_name}_{timestamp}_config.json'

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp,
            'backbone': model.backbone_type,
            'activation': model.activation_type
        }
        torch.save(checkpoint, model_path)

        config = {
            'model_name': model_name,
            'timestamp': timestamp,
            'train_acc': float(metrics.get('train_acc', 0)),
            'test_acc': float(metrics.get('test_acc', 0)),
            'train_loss': float(metrics.get('train_loss', 0)),
            'test_loss': float(metrics.get('test_loss', 0)),
            'epoch': epoch,
            'is_final': metrics.get('is_final', False),
            'lambda_contra': float(metrics.get('lambda_contra', 0)),
            'dropout_rate': float(metrics.get('dropout_rate', 0.5)),
            'backbone': model.backbone_type,
            'activation': model.activation_type
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        is_final_str = " [FINAL]" if metrics.get('is_final', False) else ""
        print(f"\n✓ 模型已保存: {model_path}{is_final_str}")
        print(f"  激活函数: {model.activation_type}")
        return model_path, config_path

    def load_model(self, model, device, model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = checkpoint.get('metrics', {})
        print(f"\n✓ 模型已加载: {model_path}")
        print(f"  训练epoch: {checkpoint['epoch']}")
        print(f"  Backbone: {checkpoint.get('backbone', 'N/A')}")
        print(f"  激活函数: {checkpoint.get('activation', 'N/A')}")
        train_acc = metrics.get('train_acc', 'N/A')
        test_acc = metrics.get('test_acc', 'N/A')
        train_acc_str = f"{train_acc:.4f}" if isinstance(train_acc, (int, float)) else str(train_acc)
        test_acc_str = f"{test_acc:.4f}" if isinstance(test_acc, (int, float)) else str(test_acc)
        print(f"  训练准确率: {train_acc_str}")
        print(f"  测试准确率: {test_acc_str}")
        is_final = metrics.get('is_final', False)
        if is_final:
            print(f"  【这是最后一轮模型】")
        return model

    def list_models(self):
        pth_files = list(self.model_dir.glob('*.pth'))
        if len(pth_files) == 0:
            print("❌ 没有保存的模型")
            return []

        print("\n✅ 已保存的模型 (最新10个):")

        final_models = [f for f in pth_files if '_final_' in f.name]
        best_models = [f for f in pth_files if '_final_' not in f.name]

        if final_models:
            print("\n【最后一轮模型】")
            for pth_file in sorted(final_models, reverse=True)[:5]:
                config_file = pth_file.with_suffix('')
                config_file = pth_file.parent / (config_file.name + '_config.json')

                print(f"\n{pth_file.name}")

                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    test_acc = config.get('test_acc', 'N/A')
                    test_acc_str = f"{test_acc:.4f}" if isinstance(test_acc, (int, float)) else str(test_acc)
                    print(f"  测试准确率: {test_acc_str}")
                    print(f"  时间: {config.get('timestamp', 'N/A')}")
                    print(f"  Epoch: {config.get('epoch', 'N/A')}")
                    print(f"  Backbone: {config.get('backbone', 'N/A')}")
                    print(f"  激活函数: {config.get('activation', 'N/A')}")

        if best_models:
            print("\n【最佳模型】")
            for pth_file in sorted(best_models, reverse=True)[:5]:
                config_file = pth_file.with_suffix('')
                config_file = pth_file.parent / (config_file.name + '_config.json')

                print(f"\n{pth_file.name}")

                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    test_acc = config.get('test_acc', 'N/A')
                    test_acc_str = f"{test_acc:.4f}" if isinstance(test_acc, (int, float)) else str(test_acc)
                    print(f"  测试准确率: {test_acc_str}")
                    print(f"  时间: {config.get('timestamp', 'N/A')}")
                    print(f"  Epoch: {config.get('epoch', 'N/A')}")
                    print(f"  Backbone: {config.get('backbone', 'N/A')}")
                    print(f"  激活函数: {config.get('activation', 'N/A')}")


# ============== 主训练函数(添加activation参数) ==============
def train_step_contrastive(step_name, train_dataloader, test_dataloader, class_names,
                           num_epochs=30, model_save_name='fadel_vit', use_frequency=False,
                           lambda_contra=0.5, dropout_rate=0.5, activation='softplus'):
    """
    结合EDL损失和对比学习损失的训练

    Args:
        lambda_contra: 对比损失权重,推荐值:
            - 0.3: 保守,主要关注分类准确率
            - 0.5: 平衡(默认)
            - 0.7: 激进,强调特征分离
        dropout_rate: Dropout比率
        activation: 证据激活函数类型 ('softplus', 'relu', 'exp')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    print("=" * 70)
    print(f"【{step_name} - 对比学习版本 (ViT-B/16)】")
    print(f"【对比损失权重: {lambda_contra}】")
    print(f"【Dropout Rate: {dropout_rate}】")
    print(f"【激活函数: {activation}】")
    print(f"【Epochs: {num_epochs}】")
    if use_frequency:
        print("【使用频域输入 + FADEL模型】")
    print("=" * 70)

    print("\n初始化模型...")
    model = FADELContrastiveModel(
        backbone='vit_b_16',
        dropout_rate=dropout_rate,
        input_channels=6 if use_frequency else 3,
        feature_dim=512,
        activation=activation
    ).to(device)
    print(f"✓ 对比学习FADEL模型 (ViT-B/16, {activation}) 已加载到GPU")

    # 两个损失函数
    criterion_edl = EDLLoss(num_classes=2, weights=[1.0, 1.0])
    criterion_contra = SupConLoss(temperature=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    patience = 10
    best_acc = 0.0
    patience_counter = 0
    model_manager = ModelManager(step_name=step_name)

    print("\n" + "=" * 70)
    print("开始训练(带对比学习)")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc, train_edl, train_contra, train_labels, train_preds = train_epoch_contrastive(
            model, train_dataloader, optimizer, criterion_edl, criterion_contra, device, lambda_contra
        )
        print(
            f"Train Loss: {train_loss:.4f} (EDL: {train_edl:.4f}, Contra: {train_contra:.4f}), Train Acc: {train_acc:.4f}")

        # 计算各类准确率
        for i, class_name in enumerate(class_names):
            mask = train_labels == i
            if mask.sum() > 0:
                class_acc = (train_preds[mask] == train_labels[mask]).sum() / mask.sum()
                print(f"  {class_name} Accuracy: {class_acc:.4f}", end="")
            if i < len(class_names) - 1:
                print(" | ", end="")
        print()

        test_result = evaluate_contrastive(model, test_dataloader, criterion_edl, criterion_contra, device)
        test_loss = test_result['loss']
        test_acc = test_result['acc']
        print(
            f"Test Loss: {test_loss:.4f} (EDL: {test_result['edl_loss']:.4f}, Contra: {test_result['contra_loss']:.4f}), Test Acc: {test_acc:.4f}")

        # 计算测试集各类准确率
        for i, class_name in enumerate(class_names):
            mask = test_result['labels'] == i
            if mask.sum() > 0:
                class_acc = (test_result['preds'][mask] == test_result['labels'][mask]).sum() / mask.sum()
                print(f"  {class_name} Accuracy: {class_acc:.4f}", end="")
            if i < len(class_names) - 1:
                print(" | ", end="")
        print()

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0

            metrics = {
                'train_acc': float(train_acc),
                'train_loss': float(train_loss),
                'test_acc': float(test_acc),
                'test_loss': float(test_loss),
                'lambda_contra': float(lambda_contra),
                'dropout_rate': float(dropout_rate),
                'activation': activation
            }
            model_manager.save_model(model, optimizer, metrics, epoch + 1, model_save_name)
            print(f"✓ 最佳模型已保存 (Test Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ Early Stopping在Epoch {epoch + 1}触发!")
                break

    print(f"\n保存最后一轮的模型...")
    metrics_final = {
        'train_acc': float(train_acc),
        'train_loss': float(train_loss),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'is_final': True,
        'lambda_contra': float(lambda_contra),
        'dropout_rate': float(dropout_rate),
        'activation': activation
    }
    model_manager.save_model(model, optimizer, metrics_final, epoch + 1, f'{model_save_name}_final')

    print("\n" + "=" * 70)
    print(f"【{step_name} - 最终评估】")
    print("=" * 70)

    model_files = list(model_manager.model_dir.glob(f'{model_save_name}_final_*.pth'))
    if not model_files:
        model_files = list(model_manager.model_dir.glob(f'{model_save_name}_*.pth'))

    if model_files:
        latest_model = sorted(model_files, reverse=True)[0]
        print(f"\n加载模型: {latest_model.name}")
        model_manager.load_model(model, device, latest_model)

        print("\n评估训练集...")
        train_result = evaluate_contrastive(model, train_dataloader, criterion_edl, criterion_contra, device)

        print("\n评估测试集...")
        test_result = evaluate_contrastive(model, test_dataloader, criterion_edl, criterion_contra, device)

        analyze_separation(train_result['features'], train_result['labels'], f'{step_name} - 训练集')
        analyze_separation(test_result['features'], test_result['labels'], f'{step_name} - 测试集')

        print("\n" + "=" * 70)
        print(f"【{step_name}训练集分类报告】")
        print("=" * 70)
        print_custom_classification_report(train_result['labels'], train_result['preds'],
                                           train_result['probs'], class_names)

        print("\n" + "=" * 70)
        print(f"【{step_name}测试集分类报告】")
        print("=" * 70)
        print_custom_classification_report(test_result['labels'], test_result['preds'],
                                           test_result['probs'], class_names)

        print("\n绘制对比图...")
        save_name = f'fadel_vit_{step_name.replace(" ", "_").replace(":", "")}'
        plot_train_test_comparison(train_result, test_result, class_names, step_name + ' - ', save_name)

        print(f"\n✓ {step_name}完成!")


# ============== 测试函数 ==============
def test_with_trained_model(step_name, test_dataloader, class_names, model_save_name='fadel_vit',
                            use_frequency=False, train_dataloader=None, activation='softplus'):
    """
    使用训练好的模型进行测试

    Args:
        step_name: 步骤名称
        test_dataloader: 测试数据加载器
        class_names: 类名列表
        model_save_name: 模型名称前缀
        use_frequency: 是否使用频域输入
        train_dataloader: 可选的训练集加载器,用于对比绘图
        activation: 证据激活函数类型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    print("=" * 70)
    print(f"【{step_name} - 测试阶段 (ViT-B/16)】")
    print("=" * 70)

    print("\n初始化模型...")
    model = FADELContrastiveModel(
        backbone='vit_b_16',
        dropout_rate=0.5,
        input_channels=6 if use_frequency else 3,
        feature_dim=512,
        activation=activation
    ).to(device)
    print(f"✓ 模型 (ViT-B/16, {activation}) 已加载到GPU")

    # 损失函数
    criterion_edl = EDLLoss(num_classes=2, weights=[1.0, 1.0])
    criterion_contra = SupConLoss(temperature=0.1)

    model_manager = ModelManager(step_name=step_name)

    print("\n查找已保存的模型...")
    # 优先查找final模型
    model_files = list(model_manager.model_dir.glob(f'{model_save_name}_final_*.pth'))
    if not model_files:
        model_files = list(model_manager.model_dir.glob(f'{model_save_name}_*.pth'))

    if not model_files:
        print(f"❌ 未找到模型: {model_save_name}")
        return None

    latest_model = sorted(model_files, reverse=True)[0]
    print(f"✓ 加载模型: {latest_model.name}")
    model_manager.load_model(model, device, latest_model)

    print("\n" + "=" * 70)
    print(f"【{step_name} - 测试集评估】")
    print("=" * 70)

    print("\n评估测试集...")
    test_result = evaluate_contrastive(model, test_dataloader, criterion_edl, criterion_contra, device)

    print(f"\n测试集整体性能:")
    print(f"  Loss: {test_result['loss']:.4f}")
    print(f"  Accuracy: {test_result['acc']:.4f}")

    # 计算各类准确率
    for i, class_name in enumerate(class_names):
        mask = test_result['labels'] == i
        if mask.sum() > 0:
            class_acc = (test_result['preds'][mask] == test_result['labels'][mask]).sum() / mask.sum()
            print(f"  {class_name} Accuracy: {class_acc:.4f}")

    print("\n" + "=" * 70)
    print(f"【{step_name} - 详细分类报告】")
    print("=" * 70)
    print_custom_classification_report(test_result['labels'], test_result['preds'],
                                       test_result['probs'], class_names)

    # 特征分离度分析
    analyze_separation(test_result['features'], test_result['labels'], f'{step_name}')

    # 如果提供了训练数据,进行对比绘图
    if train_dataloader is not None:
        print("\n评估训练集(用于对比分析)...")
        train_result = evaluate_contrastive(model, train_dataloader, criterion_edl, criterion_contra, device)

        print("\n绘制训练集vs测试集对比图...")
        save_name = f'fadel_vit_{step_name.replace(" ", "_").replace("：", "").replace(":", "")}'
        plot_train_test_comparison(train_result, test_result, class_names, step_name + ' - ', save_name)
    else:
        print("\n⚠️ 未提供训练集数据,跳过对比绘图")

    print(f"\n✓ {step_name}测试完成!\n")
    return test_result


# ============== 修改后的main函数 - 支持单独训练步骤和激活函数切换 ==============
def main():
    """
    两步分类流程 - 可选训练或测试,支持单独训练步骤和激活函数切换
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    train_gan_path = pathlib.Path(r'E:\New folder3\gan\StyleGAN')
    train_dm_path = pathlib.Path(r'E:\New folder3\dm\ADM-20251016T055152Z-1-001\ADM')
    train_real_path = pathlib.Path(r'E:\数据集\real\00000')

    test_gan_path = pathlib.Path(r'E:\New folder3\gan\StyleGAN2-20251016T055258Z-1-001\StyleGAN2')
    test_dm_path = pathlib.Path(r'E:\New folder3\dm\IDDPM-20251016T055256Z-1-001\IDDPM')
    test_real_path = pathlib.Path(r'E:\数据集\real\00000')

    transform_train, transform_test = get_transforms()

    # ============== 步骤1和步骤2的配置参数 ==============
    STEP1_EPOCHS = 50
    STEP1_DROPOUT = 0.2
    STEP1_LAMBDA_CONTRA = 0.5
    STEP1_ACTIVATION = 'relu'  # 'softplus', 'relu', 'exp'

    STEP2_EPOCHS = 50
    STEP2_DROPOUT = 0.2
    STEP2_LAMBDA_CONTRA = 0.5
    STEP2_ACTIVATION = 'relu'  # 'softplus', 'relu', 'exp'

    print("\n" + "=" * 70)
    print("【选择操作模式】")
    print("=" * 70)
    print("1. train       - 训练两个步骤")
    print("2. train1      - 只训练步骤1 (Real vs Fake)")
    print("3. train2      - 只训练步骤2 (GAN vs Diffusion)")
    print("4. test1       - 只测试步骤1 (Real vs Fake)")
    print("5. test2       - 只测试步骤2 (GAN vs Diffusion)")
    print("6. test_both   - 依次测试步骤1和步骤2")

    mode = input("\n请输入模式 (train/train1/train2/test1/test2/test_both): ").strip().lower()

    if mode == 'train' or mode == 'train1':
        # ============== 训练步骤1 ==============
        print("\n" + "=" * 70)
        print("【步骤1: Real vs Fake 二分类 - 对比学习 (ViT-B/16)】")
        print("=" * 70)

        print("\n加载步骤1数据集...")
        train_dataset_step1 = RealFakeDataset(train_gan_path, train_dm_path, train_real_path,
                                              transform=transform_train, max_samples=1000)
        train_loader_step1 = DataLoader(train_dataset_step1, batch_size=16, shuffle=True, num_workers=0)

        test_dataset_step1 = RealFakeDataset(test_gan_path, test_dm_path, test_real_path,
                                             transform=transform_test, max_samples=500)
        test_loader_step1 = DataLoader(test_dataset_step1, batch_size=16, shuffle=False, num_workers=0)

        train_labels_array = np.array(train_dataset_step1.labels)
        test_labels_array = np.array(test_dataset_step1.labels)

        print(f"✓ 训练集: {len(train_dataset_step1)} (Fake: {(train_labels_array == 0).sum()}, "
              f"Real: {(train_labels_array == 1).sum()})")
        print(f"✓ 测试集: {len(test_dataset_step1)} (Fake: {(test_labels_array == 0).sum()}, "
              f"Real: {(test_labels_array == 1).sum()})")

        train_step_contrastive(
            step_name='步骤1: Real vs Fake',
            train_dataloader=train_loader_step1,
            test_dataloader=test_loader_step1,
            class_names=['Fake', 'Real'],
            num_epochs=STEP1_EPOCHS,
            model_save_name='fadel_vit_step1_contra',
            use_frequency=False,
            lambda_contra=STEP1_LAMBDA_CONTRA,
            dropout_rate=STEP1_DROPOUT,
            activation=STEP1_ACTIVATION
        )

        if mode == 'train1':
            print("\n" + "=" * 70)
            print("✓ 步骤1训练完成!")
            print("=" * 70)
            return

    if mode == 'train' or mode == 'train2':
        # ============== 训练步骤2 ==============
        print("\n" + "=" * 70)
        print("【步骤2: GAN vs Diffusion 二分类 - 对比学习 (ViT-B/16)】")
        print("=" * 70)

        print("\n加载步骤2数据集...")
        train_dataset_step2 = GanDmDataset(train_gan_path, train_dm_path,
                                           transform=transform_train, max_samples=1000)
        train_dataset_step2 = FrequencyDatasetWrapper(train_dataset_step2, convert_to_frequency=True)
        train_loader_step2 = DataLoader(train_dataset_step2, batch_size=16, shuffle=True, num_workers=0)

        test_dataset_step2 = GanDmDataset(test_gan_path, test_dm_path,
                                          transform=transform_test, max_samples=1000)
        test_dataset_step2 = FrequencyDatasetWrapper(test_dataset_step2, convert_to_frequency=True)
        test_loader_step2 = DataLoader(test_dataset_step2, batch_size=16, shuffle=False, num_workers=0)

        train_dataset_step2_orig = GanDmDataset(train_gan_path, train_dm_path,
                                                transform=transform_train, max_samples=1000)
        test_dataset_step2_orig = GanDmDataset(test_gan_path, test_dm_path,
                                               transform=transform_test, max_samples=1000)

        train_labels_array = np.array(train_dataset_step2_orig.labels)
        test_labels_array = np.array(test_dataset_step2_orig.labels)

        print(f"✓ 训练集: {len(train_dataset_step2)} (GAN: {(train_labels_array == 0).sum()}, "
              f"Diffusion: {(train_labels_array == 1).sum()})")
        print(f"✓ 测试集: {len(test_dataset_step2)} (GAN: {(test_labels_array == 0).sum()}, "
              f"Diffusion: {(test_labels_array == 1).sum()})")

        train_step_contrastive(
            step_name='步骤2 GAN vs Diffusion',
            train_dataloader=train_loader_step2,
            test_dataloader=test_loader_step2,
            class_names=['GAN', 'Diffusion'],
            num_epochs=STEP2_EPOCHS,
            model_save_name='fadel_vit_step2_contra_freq',
            use_frequency=True,
            lambda_contra=STEP2_LAMBDA_CONTRA,
            dropout_rate=STEP2_DROPOUT,
            activation=STEP2_ACTIVATION
        )

        if mode == 'train2':
            print("\n" + "=" * 70)
            print("✓ 步骤2训练完成!")
            print("=" * 70)
            return

    if mode == 'train':
        print("\n" + "=" * 70)
        print("✓ 两步分类流程完成!")
        print("=" * 70)

    elif mode == 'test1':
        # ============== 只测试步骤1 ==============
        print("\n" + "=" * 70)
        print("【步骤1: Real vs Fake - 只测试 (ViT-B/16)】")
        print("=" * 70)

        print("\n加载训练集(用于对比分析)...")
        train_dataset_step1 = RealFakeDataset(train_gan_path, train_dm_path, train_real_path,
                                              transform=transform_train, max_samples=1000)
        train_loader_step1 = DataLoader(train_dataset_step1, batch_size=16, shuffle=False, num_workers=0)
        train_labels_array = np.array(train_dataset_step1.labels)
        print(f"✓ 训练集: {len(train_dataset_step1)} (Fake: {(train_labels_array == 0).sum()}, "
              f"Real: {(train_labels_array == 1).sum()})")

        print("\n加载测试集...")
        test_dataset_step1 = RealFakeDataset(test_gan_path, test_dm_path, test_real_path,
                                             transform=transform_test, max_samples=500)
        test_loader_step1 = DataLoader(test_dataset_step1, batch_size=16, shuffle=False, num_workers=0)

        test_labels_array = np.array(test_dataset_step1.labels)
        print(f"✓ 测试集: {len(test_dataset_step1)} (Fake: {(test_labels_array == 0).sum()}, "
              f"Real: {(test_labels_array == 1).sum()})")

        test_with_trained_model(
            step_name='步骤1: Real vs Fake',
            test_dataloader=test_loader_step1,
            class_names=['Fake', 'Real'],
            model_save_name='fadel_vit_step1_contra',
            use_frequency=False,
            train_dataloader=train_loader_step1,
            activation=STEP1_ACTIVATION
        )

        print("\n" + "=" * 70)
        print("✓ 步骤1测试完成!")
        print("=" * 70)

    elif mode == 'test2':
        # ============== 只测试步骤2 ==============
        print("\n" + "=" * 70)
        print("【步骤2: GAN vs Diffusion - 只测试 (ViT-B/16)】")
        print("=" * 70)

        print("\n加载训练集(用于对比分析)...")
        train_dataset_step2 = GanDmDataset(train_gan_path, train_dm_path,
                                           transform=transform_train, max_samples=1000)
        train_dataset_step2 = FrequencyDatasetWrapper(train_dataset_step2, convert_to_frequency=True)
        train_loader_step2 = DataLoader(train_dataset_step2, batch_size=16, shuffle=False, num_workers=0)

        train_dataset_step2_orig = GanDmDataset(train_gan_path, train_dm_path,
                                                transform=transform_train, max_samples=1000)
        train_labels_array = np.array(train_dataset_step2_orig.labels)
        print(f"✓ 训练集: {len(train_dataset_step2)} (GAN: {(train_labels_array == 0).sum()}, "
              f"Diffusion: {(train_labels_array == 1).sum()})")

        print("\n加载测试集...")
        test_dataset_step2 = GanDmDataset(test_gan_path, test_dm_path,
                                          transform=transform_test, max_samples=1000)
        test_dataset_step2 = FrequencyDatasetWrapper(test_dataset_step2, convert_to_frequency=True)
        test_loader_step2 = DataLoader(test_dataset_step2, batch_size=16, shuffle=False, num_workers=0)

        test_dataset_step2_orig = GanDmDataset(test_gan_path, test_dm_path,
                                               transform=transform_test, max_samples=1000)
        test_labels_array = np.array(test_dataset_step2_orig.labels)
        print(f"✓ 测试集: {len(test_dataset_step2)} (GAN: {(test_labels_array == 0).sum()}, "
              f"Diffusion: {(test_labels_array == 1).sum()})")

        test_with_trained_model(
            step_name='步骤2: GAN vs Diffusion',
            test_dataloader=test_loader_step2,
            class_names=['GAN', 'Diffusion'],
            model_save_name='fadel_vit_step2_contra_freq',
            use_frequency=True,
            train_dataloader=train_loader_step2,
            activation=STEP2_ACTIVATION
        )

        print("\n" + "=" * 70)
        print("✓ 步骤2测试完成!")
        print("=" * 70)

    elif mode == 'test_both':
        # ============== 测试步骤1和步骤2 ==============
        print("\n" + "=" * 70)
        print("【步骤1: Real vs Fake - 测试 (ViT-B/16)】")
        print("=" * 70)

        print("\n加载训练集(用于对比分析)...")
        train_dataset_step1 = RealFakeDataset(train_gan_path, train_dm_path, train_real_path,
                                              transform=transform_train, max_samples=1000)
        train_loader_step1 = DataLoader(train_dataset_step1, batch_size=16, shuffle=False, num_workers=0)
        train_labels_array = np.array(train_dataset_step1.labels)
        print(f"✓ 训练集: {len(train_dataset_step1)} (Fake: {(train_labels_array == 0).sum()}, "
              f"Real: {(train_labels_array == 1).sum()})")

        print("\n加载测试集...")
        test_dataset_step1 = RealFakeDataset(test_gan_path, test_dm_path, test_real_path,
                                             transform=transform_test, max_samples=500)
        test_loader_step1 = DataLoader(test_dataset_step1, batch_size=16, shuffle=False, num_workers=0)

        test_labels_array = np.array(test_dataset_step1.labels)
        print(f"✓ 测试集: {len(test_dataset_step1)} (Fake: {(test_labels_array == 0).sum()}, "
              f"Real: {(test_labels_array == 1).sum()})")

        test_with_trained_model(
            step_name='步骤1: Real vs Fake',
            test_dataloader=test_loader_step1,
            class_names=['Fake', 'Real'],
            model_save_name='fadel_vit_step1_contra',
            use_frequency=False,
            train_dataloader=train_loader_step1,
            activation=STEP1_ACTIVATION
        )

        print("\n" + "=" * 70)
        print("【步骤2: GAN vs Diffusion - 测试 (ViT-B/16)】")
        print("=" * 70)

        print("\n加载训练集(用于对比分析)...")
        train_dataset_step2 = GanDmDataset(train_gan_path, train_dm_path,
                                           transform=transform_train, max_samples=1000)
        train_dataset_step2 = FrequencyDatasetWrapper(train_dataset_step2, convert_to_frequency=True)
        train_loader_step2 = DataLoader(train_dataset_step2, batch_size=16, shuffle=False, num_workers=0)

        train_dataset_step2_orig = GanDmDataset(train_gan_path, train_dm_path,
                                                transform=transform_train, max_samples=1000)
        train_labels_array = np.array(train_dataset_step2_orig.labels)
        print(f"✓ 训练集: {len(train_dataset_step2)} (GAN: {(train_labels_array == 0).sum()}, "
              f"Diffusion: {(train_labels_array == 1).sum()})")

        print("\n加载测试集...")
        test_dataset_step2 = GanDmDataset(test_gan_path, test_dm_path,
                                          transform=transform_test, max_samples=1000)
        test_dataset_step2 = FrequencyDatasetWrapper(test_dataset_step2, convert_to_frequency=True)
        test_loader_step2 = DataLoader(test_dataset_step2, batch_size=16, shuffle=False, num_workers=0)

        test_dataset_step2_orig = GanDmDataset(test_gan_path, test_dm_path,
                                               transform=transform_test, max_samples=1000)
        test_labels_array = np.array(test_dataset_step2_orig.labels)
        print(f"✓ 测试集: {len(test_dataset_step2)} (GAN: {(test_labels_array == 0).sum()}, "
              f"Diffusion: {(test_labels_array == 1).sum()})")

        test_with_trained_model(
            step_name='步骤2: GAN vs Diffusion',
            test_dataloader=test_loader_step2,
            class_names=['GAN', 'Diffusion'],
            model_save_name='fadel_vit_step2_contra_freq',
            use_frequency=True,
            train_dataloader=train_loader_step2,
            activation=STEP2_ACTIVATION
        )

        print("\n" + "=" * 70)
        print("✓ 两步骤测试完成!")
        print("=" * 70)

    else:
        print("❌ 无效的模式!请输入 train、train1、train2、test1、test2 或 test_both")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode in ['train', 'train1', 'train2']:
            import builtins

            original_input = builtins.input
            builtins.input = lambda x: mode
            main()
            builtins.input = original_input

        elif mode == 'test1':
            import builtins

            original_input = builtins.input
            builtins.input = lambda x: 'test1'
            main()
            builtins.input = original_input

        elif mode == 'test2':
            import builtins

            original_input = builtins.input
            builtins.input = lambda x: 'test2'
            main()
            builtins.input = original_input

        elif mode == 'test_both':
            import builtins

            original_input = builtins.input
            builtins.input = lambda x: 'test_both'
            main()
            builtins.input = original_input

        elif mode == 'list':
            print("=" * 70)
            print("【步骤1: Real vs Fake - 已保存的模型 (ViT)】")
            print("=" * 70)
            ModelManager(step_name='步骤1: Real vs Fake').list_models()

            print("\n" + "=" * 70)
            print("【步骤2: GAN vs Diffusion - 已保存的模型 (ViT)】")
            print("=" * 70)
            ModelManager(step_name='步骤2 GAN vs Diffusion').list_models()

        else:
            print("❌ 无效的命令!")
            print("可用命令: train, train1, train2, test1, test2, test_both, list")

    else:
        main()