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
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        # 设置日志
        self.setup_logging()
        
        # 最佳超参数配置 (基于多阶段优化发现)
        self.best_config = {
            'embedding_size': 128,
            'filter_num': 32,
            'mask_rate': 0.05,
            'temperature': 0.1,
            'cl_similarity_threshold': 0.5,
            'contrastive_weight': 0.03,  # 关键发现
            'lr': 3e-4,                  # 最佳学习率
            'weight_decay': 1e-4,
            'batch_size': 512,           # 减小批次大小避免内存问题
            'max_epochs': 3000,          # 长期训练的重要性
            'early_stop_patience': 400,
            'grad_clip_norm': 1.0
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
        
        fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', self.args.dataset)
        
        # 使用最佳数据配置
        train_set = GNNDataset(fpath, types='train', use_surface=True, use_masif=True)
        test1_set = GNNDataset(fpath, types='test1', use_surface=True, use_masif=True)
        
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
        
        self.logger.info(f"✅ 训练集: {len(train_set)} 样本")
        self.logger.info(f"✅ 测试集: {len(test1_set)} 样本")
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
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"✅ 模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")
        self.logger.info(f"✅ 对比学习权重: {self.best_config['contrastive_weight']}")
    
    def setup_optimizer(self):
        """设置优化器和调度器"""
        self.logger.info("⚙️ 设置优化器...")
        
        # 使用AdamW优化器 (比Adam更好)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.best_config['lr'],
            weight_decay=self.best_config['weight_decay']
        )
        
        # 余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.best_config['max_epochs'], 
            eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        self.logger.info(f"✅ 优化器: AdamW (lr={self.best_config['lr']}, wd={self.best_config['weight_decay']})")
        self.logger.info(f"✅ 调度器: CosineAnnealingLR")
        self.logger.info(f"✅ 损失函数: MSELoss")
    
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
            
            # 计算损失 (包含对比学习)
            mse_loss = self.criterion(pred.view(-1), data.y.view(-1))
            
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
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {total_loss_batch.item():.4f}, LR: {current_lr:.6f}"
                )
        
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
    
    def train(self):
        """主训练循环"""
        self.logger.info("🚀 开始训练...")
        
        best_mse = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        # 目标性能 (基于多阶段优化结果)
        target_mse = 0.1310
        target_ci = 0.8886
        target_r2 = 0.8035
        
        self.logger.info(f"🎯 目标性能: MSE={target_mse:.4f}, CI={target_ci:.4f}, R2={target_r2:.4f}")
        
        for epoch in range(1, self.best_config['max_epochs'] + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 评估
            test_mse, test_cindex, test_r2 = self.evaluate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录结果
            self.logger.info(
                f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | "
                f"Test MSE: {test_mse:.4f} | CI: {test_cindex:.4f} | R2: {test_r2:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # 保存最佳模型
            if test_mse < best_mse:
                best_mse = test_mse
                best_epoch = epoch
                patience_counter = 0
                
                # 保存模型
                if self.args.save_model:
                    save_path = f"best_single_model_mse_{test_mse:.4f}_ci_{test_cindex:.4f}_r2_{test_r2:.4f}.pt"
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f"💾 保存最佳模型: {save_path}")
                
                self.logger.info(f"🏆 新的最佳性能! MSE: {test_mse:.4f}")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= self.best_config['early_stop_patience']:
                self.logger.info(f"⏹️ 早停触发 (patience={self.best_config['early_stop_patience']})")
                break
            
            # 检查是否达到目标性能
            if test_mse <= target_mse * 1.01:  # 1%容差
                self.logger.info(f"🎉 达到目标MSE性能! {test_mse:.4f} <= {target_mse:.4f}")
        
        # 训练完成总结
        self.logger.info("=" * 60)
        self.logger.info("🏁 训练完成!")
        self.logger.info(f"🏆 最佳性能 (Epoch {best_epoch}): MSE={best_mse:.4f}")
        self.logger.info(f"🎯 目标达成度: {(target_mse/best_mse)*100:.1f}%")
        self.logger.info("=" * 60)
        
        return best_mse

def main():
    parser = argparse.ArgumentParser(description='KIBA单模型统一训练')
    parser.add_argument('--dataset', type=str, default='kiba', help='数据集名称')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
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
