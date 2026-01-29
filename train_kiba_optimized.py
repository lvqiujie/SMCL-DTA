#!/usr/bin/env python3
"""
优化版KIBA训练脚本 - 解决性能问题
基于Step 2最佳配置 + 轻量级增强
"""

import os
import math
import numpy as np
import torch.optim as optim
import torch
import warnings
import random

warnings.filterwarnings('ignore', message='TypedStorage is deprecated')
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse

from metrics import get_cindex, get_rm2
from dataset import *
from model_0428_16_dual import MGraphDTA
from utils import *
from log.train_logger import TrainLogger

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    # 🔥 始终使用MSE作为评估指标
    mse_criterion = nn.MSELoss()

    pred_list = []
    label_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            # 训练损失（可能是其他类型）
            train_loss = criterion(pred.view(-1), data.y.view(-1))
            # 评估损失（始终是MSE）
            eval_loss = mse_criterion(pred.view(-1), data.y.view(-1))

            label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(eval_loss.item(), label.size(0))  # 使用MSE评估

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    epoch_cindex = get_cindex(label, pred)
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()  # 这是MSE
    running_loss.reset()

    model.train()

    return epoch_loss, epoch_cindex, epoch_r2

class LabelSmoothingLoss(nn.Module):
    """轻量级标签平滑损失"""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        # 简单的平滑正则化
        pred_mean = pred.mean()
        target_mean = target.mean()
        mean_regularization = (pred_mean - target_mean) ** 2
        return mse + self.smoothing * mean_regularization

class AdaptiveHuberLoss(nn.Module):
    """自适应Huber损失 - 对异常值更鲁棒，有助于降低MSE"""
    def __init__(self, delta=0.1, adaptive=True):
        super().__init__()
        self.delta = delta
        self.adaptive = adaptive
        self.huber_loss = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        if self.adaptive:
            # 根据预测误差动态调整delta
            residual = torch.abs(pred - target)
            current_delta = torch.quantile(residual, 0.7).item()  # 70%分位数
            self.huber_loss.delta = min(max(current_delta, 0.05), 0.3)

        return self.huber_loss(pred, target)

class FocalMSELoss(nn.Module):
    """Focal MSE损失 - 专注于困难样本，有助于精确预测"""
    def __init__(self, alpha=2.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # 计算focal权重：误差越大，权重越高
        focal_weight = (mse / (mse.mean() + 1e-8)) ** self.gamma
        focal_mse = self.alpha * focal_weight * mse
        return focal_mse.mean()

class QuantileLoss(nn.Module):
    """分位数损失 - 更好地处理标签分布的尾部"""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        losses = []
        for q in self.quantiles:
            error = target - pred
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        return torch.stack(losses).mean()

class CombinedLoss(nn.Module):
    """组合损失 - 结合多种损失的优势"""
    def __init__(self, huber_weight=0.7, focal_weight=0.2, quantile_weight=0.1):
        super().__init__()
        self.huber_loss = AdaptiveHuberLoss(delta=0.1)
        self.focal_loss = FocalMSELoss(alpha=1.5, gamma=0.5)
        self.quantile_loss = QuantileLoss()

        self.huber_weight = huber_weight
        self.focal_weight = focal_weight
        self.quantile_weight = quantile_weight

    def forward(self, pred, target):
        huber = self.huber_loss(pred, target)
        focal = self.focal_loss(pred, target)
        quantile = self.quantile_loss(pred, target)

        return (self.huber_weight * huber +
                self.focal_weight * focal +
                self.quantile_weight * quantile)

def main():
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--dataset', default='kiba', help='dataset name')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    
    # Step 2最佳配置参数
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')  # 降低学习率
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--contrastive_weight', type=float, default=0.03, help='weight for contrastive loss')
    parser.add_argument('--mask_rate', type=float, default=0.05, help='masking rate')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for contrastive loss')
    parser.add_argument('--cl_similarity_threshold', type=float, default=0.5, help='similarity threshold for regression CL')
    
    # 优化参数
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--early_stop_patience', type=int, default=150, help='early stopping patience')
    parser.add_argument('--use_adaptive_weight', action='store_true', default=True, help='use adaptive contrastive weight')

    # 🔥 损失函数选择
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['mse', 'huber', 'focal', 'quantile', 'combined', 'label_smooth'],
                       help='training loss function type')
    parser.add_argument('--huber_delta', type=float, default=0.1, help='delta for Huber loss')
    parser.add_argument('--focal_alpha', type=float, default=1.5, help='alpha for Focal MSE loss')
    parser.add_argument('--focal_gamma', type=float, default=0.5, help='gamma for Focal MSE loss')

    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        contrastive_weight=args.contrastive_weight,
        mask_rate=args.mask_rate,
        temperature=args.temperature,
        cl_similarity_threshold=args.cl_similarity_threshold,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        use_adaptive_weight=args.use_adaptive_weight,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )

    logger = TrainLogger(params)
    logger.info(__file__)
    logger.info("🚀 启动优化版KIBA训练 - 解决性能问题")

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', DATASET)

    logger.info(f"Number of train: {len(GNNDataset(fpath, types='train', use_surface=True, use_masif=True))}")
    logger.info(f"Number of test1: {len(GNNDataset(fpath, types='test1', use_surface=True, use_masif=True))}")
    
    try:
        test2_size = len(GNNDataset(fpath, types='test2', use_surface=True, use_masif=True))
        logger.info(f"Number of test2: {test2_size}")
        has_test2 = True
    except Exception as e:
        has_test2 = False

    train_set = GNNDataset(fpath, types='train', use_surface=True, use_masif=True)
    test1_set = GNNDataset(fpath, types='test1', use_surface=True, use_masif=True)
    if has_test2:
        test2_set = GNNDataset(fpath, types='test2', use_surface=True, use_masif=True)

    # 🔥 使用标准DataLoader（高效）
    logger.info("✅ 使用优化的标准DataLoader - 确保训练效率")
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8)
    test1_loader = DataLoader(test1_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)
    if has_test2:
        test2_loader = DataLoader(test2_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)

    # 设备配置
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda = '0'
    elif DATASET.lower() == 'kiba':
        cuda = '2'
    else:
        cuda = '0'
    
    device = torch.device(f"cuda:{cuda}")
    logger.info(f"数据集: {DATASET}, 使用CUDA设备: {cuda}")

    # 创建模型
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1,
                     mask_rate=params['mask_rate'],
                     temperature=params['temperature'],
                     disable_masking=False,
                     cl_mode='regression',
                     cl_similarity_threshold=params['cl_similarity_threshold'],
                     use_surface=True).to(device)

    # 设置对比学习
    model.use_contrastive = True
    model.contrastive_weight = params['contrastive_weight']

    epochs = 1500  # 增加训练轮数
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

    # 🔥 损失函数选择
    loss_type = params.get('loss_type', 'combined')
    if loss_type == 'mse':
        criterion = nn.MSELoss()
        logger.info("✅ 使用标准MSE损失")
    elif loss_type == 'huber':
        criterion = AdaptiveHuberLoss(delta=params.get('huber_delta', 0.1))
        logger.info(f"✅ 使用自适应Huber损失 (delta={params.get('huber_delta', 0.1)})")
    elif loss_type == 'focal':
        criterion = FocalMSELoss(alpha=params.get('focal_alpha', 1.5), gamma=params.get('focal_gamma', 0.5))
        logger.info(f"✅ 使用Focal MSE损失 (alpha={params.get('focal_alpha', 1.5)}, gamma={params.get('focal_gamma', 0.5)})")
    elif loss_type == 'quantile':
        criterion = QuantileLoss()
        logger.info("✅ 使用分位数损失")
    elif loss_type == 'combined':
        criterion = CombinedLoss()
        logger.info("✅ 使用组合损失 (Huber + Focal + Quantile)")
    elif loss_type == 'label_smooth':
        criterion = LabelSmoothingLoss(smoothing=0.05)
        logger.info("✅ 使用标签平滑损失")
    else:
        criterion = nn.MSELoss()
        logger.info("✅ 默认使用MSE损失")

    # 优化的优化器 - 分组学习率
    regular_params = []
    contrastive_params = []
    
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['projection', 'temperature']):
            contrastive_params.append(param)
        else:
            regular_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': regular_params, 'lr': params['lr'], 'weight_decay': params['weight_decay']},
        {'params': contrastive_params, 'lr': params['lr'] * 0.5, 'weight_decay': params['weight_decay'] * 0.5}
    ], betas=(0.9, 0.999), eps=1e-8)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    logger.info(f"🎯 目标性能: MSE < 0.13, CI > 0.9, R2 > 0.8")
    logger.info(f"📊 当前最佳基线: MSE=0.4314, CI=0.7407, R2=0.3549")
    logger.info(f"⚡ 优化配置: lr={params['lr']}, contrastive_weight={params['contrastive_weight']}")

    global_step = 0
    global_epoch = 0
    break_flag = False

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_r2 = AverageMeter()
    running_best_mse1 = BestMeter("min")

    model.train()

    logger.info("🚀 开始优化训练...")

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1
            data = data.to(device)

            # 对比学习训练
            if model.use_contrastive:
                try:
                    embeddings1 = model.get_embeddings(data, apply_masking=True)
                    embeddings2 = model.get_embeddings(data, apply_masking=True)
                    fused_embeddings = model.fuse_embeddings(embeddings1, embeddings2, strategy='average')
                    pred = model.classifier(fused_embeddings)

                    proj_embeddings1 = model.projection_head(embeddings1)
                    proj_embeddings2 = model.projection_head(embeddings2)
                    
                    mse_loss = criterion(pred.view(-1), data.y.view(-1))
                    
                    # 🔥 轻量级自适应权重
                    if params['use_adaptive_weight']:
                        label_std = data.y.std().item()
                        # 根据标签分布调整权重：分布越分散，权重越小
                        adaptive_factor = min(1.0, 1.0 / (1.0 + label_std * 0.1))
                        current_cl_weight = params['contrastive_weight'] * adaptive_factor
                    else:
                        current_cl_weight = model.contrastive_weight
                    
                    cl_loss = model.compute_contrastive_loss(proj_embeddings1, proj_embeddings2)
                    loss = mse_loss + current_cl_weight * cl_loss
                    
                except Exception as e:
                    logger.warning(f"对比学习计算出错: {e}, 回退到仅使用MSE损失")
                    pred = model(data, apply_masking=False)
                    loss = criterion(pred.view(-1), data.y.view(-1))
            else:
                pred = model(data)
                loss = criterion(pred.view(-1), data.y.view(-1))

            # 计算评估指标
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))
            epoch_r2 = get_rm2(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0))
            running_cindex.update(cindex, data.y.size(0))
            running_r2.update(epoch_r2, data.y.size(0))

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                epoch_r2 = running_r2.get_average()

                running_loss.reset()
                running_cindex.reset()
                running_r2.reset()

                # 验证
                test1_loss, test1_cindex, test1_r2 = val(model, criterion, test1_loader, device)
                if has_test2:
                    test2_loss, test2_cindex, test2_r2 = val(model, criterion, test2_loader, device)

                # 学习率调度
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                msg = f"epoch-{global_epoch}, LR-{current_lr:.6f}, MSEloss-{epoch_loss:.4f}, cindex-{epoch_cindex:.4f}, " \
                      f"r2-{epoch_r2:.4f}, test1: [MSEloss-{test1_loss:.4f}, cindex:{test1_cindex:.4f}, r2:{test1_r2:.4f}]"
                if has_test2:
                    msg = msg + f", test2: [MSEloss-{test2_loss:.4f}, cindex:{test2_cindex:.4f}, r2:{test2_r2:.4f}]"
                
                # 🎯 目标检查
                if test1_loss < 0.13 and test1_cindex > 0.9 and test1_r2 > 0.8:
                    logger.info(f"🎉 达到目标性能! {msg}")
                
                logger.info(msg)

                # 早停检查
                if test1_loss < running_best_mse1.get_best():
                    running_best_mse1.update(test1_loss)
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), msg)
                else:
                    count = running_best_mse1.counter()
                    if count > params['early_stop_patience']:
                        logger.info(f"🛑 早停在epoch {global_epoch}")
                        break_flag = True
                        break

    logger.info("🎊 优化训练完成！")

if __name__ == '__main__':
    main()
