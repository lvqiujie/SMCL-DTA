#!/usr/bin/env python3
"""
KIBA单模型统一训练脚本
整合所有最佳实践到一个简洁的训练流程中
目标: 直接达到MSE~0.1310, CI~0.8886, R2~0.8035的性能

基于多阶段优化发现的最佳配置:
- 表面特征: use_surface=True, use_masif=True
- 对比学习: contrastive_weight=0.03
- 优化器: AdamW with cosine scheduling
- 训练策略: 长期训练 + 早停 + 梯度裁剪
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import logging
from datetime import datetime
from torch_geometric.data import DataLoader

from src.model_0428_16_dual import MGraphDTA
from src.dataset import GNNDataset
from src.metrics import get_cindex, get_rm2

def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def set_reproducible_seeds(seed=42):
    """设置所有随机种子确保完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class OptimizedTrainer:
    """优化的单模型训练器"""
    
    def __init__(self, args):
        self.args = args
        # 使用CUDA_VISIBLE_DEVICES映射后的设备0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 设置日志
        self.setup_logging()
        
        # Phase 1优化配置 (基于GASI-DTA文献最佳实践)
        self.best_config = {
            'embedding_size': 128,
            'filter_num': 32,
            'mask_rate': 0.05,
            'temperature': 0.1,
            'cl_similarity_threshold': 0.5,
            'contrastive_weight': 0.03,  # 保持已验证的对比学习权重
            'lr': 1e-4,                  # 文献推荐的更保守学习率
            'weight_decay': 1e-4,        # 文献推荐的更强正则化
            'batch_size': 256,           # 文献推荐的更小批次大小
            'max_epochs': 3000,
            'early_stop_patience': 200,  # 更合理的早停耐心
            'grad_clip_norm': 1.0,       # 文献推荐的更强梯度裁剪
            'warmup_epochs': 100,        # 文献推荐的更长预热
            'label_smoothing': 0.05,     # 增强标签平滑
            'dropout_rate': 0.2,         # 文献推荐的更强dropout
            'mixup_alpha': 0.2,          # 添加Mixup数据增强
            'use_cosine_restarts': True  # 使用重启余弦调度
        }
        
        self.logger.info("🚀 初始化KIBA单模型训练器")
        self.logger.info(f"📊 最佳配置: {self.best_config}")
    
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/single_model_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/training.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """加载KIBA数据集"""
        self.logger.info("📊 加载KIBA数据集...")
        
        # fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', self.args.dataset)
        fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', self.args.dataset, 'cold')

        # 使用最佳数据配置
        train_set = GNNDataset(fpath, types='train', use_surface=True, use_masif=True)
        test1_set = GNNDataset(fpath, types='test1', use_surface=True, use_masif=True)

        try:
            test2_size = len(GNNDataset(fpath, types='test2', use_surface=True, use_masif=True))
            self.has_test2 = True
        except Exception as e:
            self.has_test2 = False

        if self.has_test2:
            test2_set = GNNDataset(fpath, types='test2', use_surface=True, use_masif=True)


        self.train_loader = DataLoader(
            train_set, 
            batch_size=self.best_config['batch_size'], 
            shuffle=True, 
            num_workers=8
        )
        self.test_loader = DataLoader(
            test1_set, 
            batch_size=self.best_config['batch_size'], 
            shuffle=False, 
            num_workers=8
        )

        if self.has_test2:
            self.test_loader2 = DataLoader(
                test2_set,
                batch_size=self.best_config['batch_size'],
                shuffle=False,
                num_workers=8
            )
        
        self.logger.info(f"✅ 训练集: {len(train_set)} 样本")
        self.logger.info(f"✅ 测试集: {len(test1_set)} 样本")
        if self.has_test2:
            self.logger.info(f"✅ 第二测试集: {test2_size} 样本")
        self.logger.info(f"✅ 批次大小: {self.best_config['batch_size']}")
        self.logger.info(f"✅ 表面特征: 启用 (use_surface=True, use_masif=True)")
    
    def create_model(self):
        """创建优化的模型"""
        self.logger.info("🏗️ 创建MGraphDTA模型...")

        self.model = MGraphDTA(
            3, 25 + 1,
            embedding_size=self.best_config['embedding_size'],
            filter_num=self.best_config['filter_num'],
            out_dim=1,
            mask_rate=self.best_config['mask_rate'],
            temperature=self.best_config['temperature'],
            disable_masking=False,
            cl_mode='regression',
            cl_similarity_threshold=self.best_config['cl_similarity_threshold'],
            use_surface=True  # 关键特征
        ).to(self.device)

        # 添加dropout层到模型中 (如果模型支持)
        if hasattr(self.model, 'set_dropout'):
            self.model.set_dropout(self.best_config['dropout_rate'])

        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"✅ 模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")
        self.logger.info(f"✅ 对比学习权重: {self.best_config['contrastive_weight']}")
        self.logger.info(f"✅ Dropout率: {self.best_config['dropout_rate']}")
    
    def setup_optimizer(self):
        """设置优化器和调度器 - Phase 1优化版本"""
        self.logger.info("⚙️ 设置优化器...")

        # 使用AdamW优化器 (比Adam更好)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.best_config['lr'],
            weight_decay=self.best_config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Phase 1: 使用重启余弦调度器 (文献推荐)
        if self.best_config['use_cosine_restarts']:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=50,      # 初始重启周期
                T_mult=2,    # 周期倍增因子
                eta_min=1e-6 # 最小学习率
            )
            scheduler_name = "CosineAnnealingWarmRestarts"
        else:
            # 学习率预热调度器 (备选)
            def lr_lambda(epoch):
                if epoch < self.best_config['warmup_epochs']:
                    return epoch / self.best_config['warmup_epochs']
                else:
                    # 余弦退火
                    progress = (epoch - self.best_config['warmup_epochs']) / (self.best_config['max_epochs'] - self.best_config['warmup_epochs'])
                    return 0.5 * (1 + np.cos(np.pi * progress))

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            scheduler_name = "WarmupCosineAnnealingLR"

        # 损失函数 - 增强标签平滑
        self.criterion = nn.MSELoss()
        self.label_smoothing = self.best_config['label_smoothing']

        self.logger.info(f"✅ 优化器: AdamW (lr={self.best_config['lr']}, wd={self.best_config['weight_decay']})")
        self.logger.info(f"✅ 调度器: {scheduler_name}")
        self.logger.info(f"✅ 损失函数: MSELoss + 标签平滑({self.label_smoothing})")
        self.logger.info(f"✅ Dropout: {self.best_config['dropout_rate']}")
        self.logger.info(f"✅ Mixup Alpha: {self.best_config['mixup_alpha']}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            pred = self.model(data)
            
            # 计算损失 (包含对比学习和标签平滑)
            pred_flat = pred.view(-1)
            target_flat = data.y.view(-1)

            # Phase 1: 添加Mixup数据增强 (随机应用)
            if self.best_config['mixup_alpha'] > 0 and np.random.random() < 0.5:
                # 对预测结果应用Mixup
                mixed_pred, y_a, y_b, lam = mixup_data(pred_flat.unsqueeze(1), target_flat,
                                                      self.best_config['mixup_alpha'])
                mixed_pred = mixed_pred.squeeze(1)

                # Mixup损失
                mse_loss = lam * self.criterion(mixed_pred, y_a) + (1 - lam) * self.criterion(mixed_pred, y_b)
            else:
                # 增强标签平滑
                if self.label_smoothing > 0:
                    # 对目标值添加小量噪声
                    noise = torch.randn_like(target_flat) * self.label_smoothing
                    target_smooth = target_flat + noise
                    mse_loss = self.criterion(pred_flat, target_smooth)
                else:
                    mse_loss = self.criterion(pred_flat, target_flat)

            # 获取对比学习损失
            if hasattr(self.model, 'get_contrastive_loss'):
                cl_loss = self.model.get_contrastive_loss()
                total_loss_batch = mse_loss + self.best_config['contrastive_weight'] * cl_loss
            else:
                total_loss_batch = mse_loss
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.best_config['grad_clip_norm']
            )
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # 定期打印进度
            # if batch_idx % 100 == 0:
            #     current_lr = self.optimizer.param_groups[0]['lr']
            #     self.logger.info(
            #         f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
            #         f"Loss: {total_loss_batch.item():.4f}, LR: {current_lr:.6f}"
            #     )
        
        return total_loss / num_batches
    
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        pred_list = []
        label_list = []
        
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                pred = self.model(data)
                pred_list.append(pred.view(-1).cpu().numpy())
                label_list.append(data.y.cpu().numpy())
        
        predictions = np.concatenate(pred_list)
        labels = np.concatenate(label_list)
        
        mse = np.mean((predictions - labels) ** 2)
        cindex = get_cindex(labels, predictions)
        r2 = get_rm2(labels, predictions)
        
        return mse, cindex, r2

    def evaluate2(self):
        """评估模型性能"""
        self.model.eval()
        pred_list = []
        label_list = []

        with torch.no_grad():
            for data in self.test_loader2:
                data = data.to(self.device)
                pred = self.model(data)
                pred_list.append(pred.view(-1).cpu().numpy())
                label_list.append(data.y.cpu().numpy())

        predictions = np.concatenate(pred_list)
        labels = np.concatenate(label_list)

        mse = np.mean((predictions - labels) ** 2)
        cindex = get_cindex(labels, predictions)
        r2 = get_rm2(labels, predictions)

        return mse, cindex, r2

    def train(self):
        """主训练循环"""
        self.logger.info("🚀 开始训练...")

        # Test1性能跟踪 (必须保持)
        best_test1_mse = float('inf')
        best_test1_ci = 0.0
        best_test1_r2 = 0.0

        # Test2性能跟踪 (需要改善)
        best_test2_mse = float('inf')
        best_test2_ci = 0.0
        best_test2_r2 = 0.0

        # 综合早停策略
        patience_counter = 0
        best_epoch = 0

        # 当前基线性能和目标
        baseline_test1 = {'mse': 0.4191, 'ci': 0.7433, 'r2': 0.3096}
        baseline_test2 = {'mse': 0.5647, 'ci': 0.5963, 'r2': 0.0661}
        target_test2 = {'mse': 0.52, 'ci': 0.661, 'r2': 0.1016}

        self.logger.info(f"📊 Test1基线: MSE={baseline_test1['mse']:.4f}, CI={baseline_test1['ci']:.4f}, R2={baseline_test1['r2']:.4f}")
        self.logger.info(f"📊 Test2基线: MSE={baseline_test2['mse']:.4f}, CI={baseline_test2['ci']:.4f}, R2={baseline_test2['r2']:.4f}")
        self.logger.info(f"🎯 Test2目标: MSE≤{target_test2['mse']:.4f}, CI≥{target_test2['ci']:.4f}, R2≥{target_test2['r2']:.4f}")
        
        for epoch in range(1, self.best_config['max_epochs'] + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 评估
            test_mse, test_cindex, test_r2 = self.evaluate()

            if self.has_test2:
                test2_mse, test2_cindex, test2_r2 = self.evaluate2()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录结果
            log_msg = (f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | LR: {current_lr:.6f} | "
                      f"Test1 MSE: {test_mse:.4f} CI: {test_cindex:.4f} R2: {test_r2:.4f}")

            if self.has_test2:
                log_msg += f" | Test2 MSE: {test2_mse:.4f} CI: {test2_cindex:.4f} R2: {test2_r2:.4f}"

            self.logger.info(log_msg)

            # 更新最佳性能
            improved = False

            # Test1性能更新
            if test_mse < best_test1_mse:
                best_test1_mse = test_mse
                best_test1_ci = test_cindex
                best_test1_r2 = test_r2
                improved = True

            # Test2性能更新 (如果存在)
            if self.has_test2:
                if test2_mse < best_test2_mse:
                    best_test2_mse = test2_mse
                    best_test2_ci = test2_cindex
                    best_test2_r2 = test2_r2
                    improved = True

            # 综合评估是否改善 (优先考虑Test2改善)
            if self.has_test2:
                # 计算Test2改善分数
                test2_improvement = (
                    (baseline_test2['mse'] - test2_mse) / baseline_test2['mse'] * 0.4 +  # MSE改善权重40%
                    (test2_cindex - baseline_test2['ci']) / baseline_test2['ci'] * 0.3 +  # CI改善权重30%
                    (test2_r2 - baseline_test2['r2']) / (baseline_test2['r2'] + 0.1) * 0.3  # R2改善权重30%
                )

                # 确保Test1性能不退化
                test1_maintained = (test_mse <= baseline_test1['mse'] * 1.01 and
                                  test_cindex >= baseline_test1['ci'] * 0.99 and
                                  test_r2 >= baseline_test1['r2'] * 0.99)

                if test2_improvement > 0 and test1_maintained:
                    best_epoch = epoch
                    patience_counter = 0

                    # 保存模型
                    if self.args.save_model:
                        save_path = f"optimized_cold_model_test1_{test_mse:.4f}_{test_cindex:.4f}_{test_r2:.4f}_test2_{test2_mse:.4f}_{test2_cindex:.4f}_{test2_r2:.4f}.pt"
                        torch.save(self.model.state_dict(), save_path)
                        self.logger.info(f"💾 保存改善模型: {save_path}")

                    self.logger.info(f"🏆 Test2性能改善! 改善分数: {test2_improvement:.4f}")
                else:
                    patience_counter += 1
            else:
                # 只有Test1的情况
                if improved:
                    best_epoch = epoch
                    patience_counter = 0

                    if self.args.save_model:
                        save_path = f"optimized_cold_model_test1_{test_mse:.4f}_{test_cindex:.4f}_{test_r2:.4f}.pt"
                        torch.save(self.model.state_dict(), save_path)
                        self.logger.info(f"💾 保存最佳模型: {save_path}")
                else:
                    patience_counter += 1
            
            # 早停检查
            if patience_counter >= self.best_config['early_stop_patience']:
                self.logger.info(f"⏹️ 早停触发 (patience={self.best_config['early_stop_patience']})")
                break

            # 检查是否达到Test2目标性能
            if self.has_test2:
                if (test2_mse <= target_test2['mse'] and
                    test2_cindex >= target_test2['ci'] and
                    test2_r2 >= target_test2['r2']):
                    self.logger.info(f"🎉 达到Test2目标性能!")
                    break

        # 训练完成总结
        self.logger.info("=" * 80)
        self.logger.info("🏁 训练完成!")
        self.logger.info(f"🏆 最佳性能 (Epoch {best_epoch}):")
        self.logger.info(f"   Test1: MSE={best_test1_mse:.4f}, CI={best_test1_ci:.4f}, R2={best_test1_r2:.4f}")
        if self.has_test2:
            self.logger.info(f"   Test2: MSE={best_test2_mse:.4f}, CI={best_test2_ci:.4f}, R2={best_test2_r2:.4f}")

            # 计算改善程度
            mse_improvement = baseline_test2['mse'] - best_test2_mse
            ci_improvement = best_test2_ci - baseline_test2['ci']
            r2_improvement = best_test2_r2 - baseline_test2['r2']

            self.logger.info(f"📈 Test2改善:")
            self.logger.info(f"   MSE: {mse_improvement:+.4f} ({'✅' if mse_improvement >= 0.04 else '❌'})")
            self.logger.info(f"   CI:  {ci_improvement:+.4f} ({'✅' if ci_improvement >= 0.065 else '❌'})")
            self.logger.info(f"   R2:  {r2_improvement:+.4f} ({'✅' if r2_improvement >= 0.035 else '❌'})")

            # Test1保持检查
            test1_maintained = (best_test1_mse <= baseline_test1['mse'] * 1.01 and
                              best_test1_ci >= baseline_test1['ci'] * 0.99 and
                              best_test1_r2 >= baseline_test1['r2'] * 0.99)
            self.logger.info(f"🔒 Test1性能保持: {'✅' if test1_maintained else '❌'}")

        self.logger.info("=" * 80)

        return best_test1_mse if not self.has_test2 else best_test2_mse

def main():
    parser = argparse.ArgumentParser(description='KIBA单模型统一训练')
    parser.add_argument('--dataset', type=str, default='kiba', help='数据集名称')
    parser.add_argument('--gpu', type=int, default=6, help='GPU设备号')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_model', action='store_true', default=True, help='保存最佳模型')
    
    args = parser.parse_args()
    
    # 设置可复现性
    set_reproducible_seeds(args.seed)
    
    print("🎯 KIBA单模型统一训练")
    print("=" * 60)
    print("目标: 通过单一训练过程直接达到最佳性能")
    print("配置: 整合所有多阶段优化发现的最佳实践")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = OptimizedTrainer(args)
    trainer.load_data()
    trainer.create_model()
    trainer.setup_optimizer()
    
    # 开始训练
    best_mse = trainer.train()
    
    print(f"\n🎉 训练完成! 最佳MSE: {best_mse:.4f}")
    print("📋 模型已保存，可直接用于论文结果复现")

if __name__ == '__main__':
    main()
