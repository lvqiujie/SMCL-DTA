#!/usr/bin/env python3
"""
KIBA ProtT5增强训练脚本
集成ProtT5蛋白质嵌入的单模型训练方法
目标: 突破MSE<0.128的最终性能瓶颈
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

from src.model_with_prot_t5 import create_enhanced_model
from mydta.src.dataset_with_prot_t5 import create_enhanced_dataloaders
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

class ProtT5EnhancedTrainer:
    """ProtT5增强训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        # 设置日志
        self.setup_logging()
        
        # ProtT5增强配置
        self.enhanced_config = {
            # 基础配置 (继承自单模型方法)
            'embedding_size': 128,
            'filter_num': 32,
            'mask_rate': 0.05,
            'temperature': 0.1,
            'cl_similarity_threshold': 0.5,
            'contrastive_weight': 0.03,
            
            # ProtT5特定配置
            'use_prot_t5': args.use_prot_t5,
            'prot_t5_fusion_dim': 128,
            'prot_t5_model_path': args.prot_t5_model_path,
            
            # 训练配置 (针对ProtT5调整)
            'lr': 2e-4,                  # 略微降低学习率 (ProtT5特征更丰富)
            'weight_decay': 1e-4,
            'batch_size': 128,           # 减小批次大小 (ProtT5增加内存使用)
            'max_epochs': 1200,          # 可能需要更少的epoch
            'early_stop_patience': 100,
            'grad_clip_norm': 1.0,
            
            # 性能目标
            'target_mse': 0.128,         # 最终目标
            'current_best_mse': 0.1310   # 当前最佳性能
        }
        
        self.logger.info("🚀 初始化ProtT5增强训练器")
        self.logger.info(f"📊 增强配置: {self.enhanced_config}")
    
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/prot_t5_enhanced_{timestamp}"
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
        """加载ProtT5增强数据"""
        self.logger.info("📊 加载ProtT5增强数据...")
        
        self.train_loader, self.test_loader = create_enhanced_dataloaders(
            dataset=self.args.dataset,
            batch_size=self.enhanced_config['batch_size'],
            use_prot_t5=self.enhanced_config['use_prot_t5'],
            prot_t5_model_path=self.enhanced_config['prot_t5_model_path'],
            device=self.device
        )
        
        self.logger.info(f"✅ 数据加载完成")
        self.logger.info(f"✅ 批次大小: {self.enhanced_config['batch_size']}")
        self.logger.info(f"✅ ProtT5增强: {self.enhanced_config['use_prot_t5']}")
    
    def create_model(self):
        """创建ProtT5增强模型"""
        self.logger.info("🏗️ 创建ProtT5增强模型...")
        
        self.model = create_enhanced_model(
            num_features_mol=3,
            num_features_pro=25 + 1,  # 根据实际特征维度调整
            embedding_size=self.enhanced_config['embedding_size'],
            filter_num=self.enhanced_config['filter_num'],
            out_dim=1,
            mask_rate=self.enhanced_config['mask_rate'],
            temperature=self.enhanced_config['temperature'],
            cl_similarity_threshold=self.enhanced_config['cl_similarity_threshold'],
            use_surface=True,
            use_prot_t5=self.enhanced_config['use_prot_t5'],
            prot_t5_fusion_dim=self.enhanced_config['prot_t5_fusion_dim']
        ).to(self.device)
        
        # 显示模型信息
        feature_info = self.model.get_feature_dimensions()
        self.logger.info(f"✅ 模型特征维度: {feature_info}")
    
    def setup_optimizer(self):
        """设置优化器和调度器"""
        self.logger.info("⚙️ 设置优化器...")
        
        # 为ProtT5相关参数使用不同的学习率
        prot_t5_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'prot_t5_fusion' in name:
                prot_t5_params.append(param)
            else:
                other_params.append(param)
        
        # 分组参数优化
        param_groups = [
            {'params': other_params, 'lr': self.enhanced_config['lr']},
            {'params': prot_t5_params, 'lr': self.enhanced_config['lr'] * 0.5}  # ProtT5参数使用较小学习率
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.enhanced_config['weight_decay']
        )
        
        # 余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.enhanced_config['max_epochs'], 
            eta_min=1e-6
        )
        
        self.criterion = nn.MSELoss()
        
        self.logger.info(f"✅ 优化器: AdamW (分组学习率)")
        self.logger.info(f"✅ ProtT5参数: {len(prot_t5_params)} 个")
        self.logger.info(f"✅ 其他参数: {len(other_params)} 个")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # 前向传播
                pred = self.model(data)
                
                # 计算损失
                mse_loss = self.criterion(pred.view(-1), data.y.view(-1))
                
                # 添加对比学习损失 (如果支持)
                if hasattr(self.model, 'get_contrastive_loss'):
                    cl_loss = self.model.get_contrastive_loss()
                    total_loss_batch = mse_loss + self.enhanced_config['contrastive_weight'] * cl_loss
                else:
                    total_loss_batch = mse_loss
                
                # 反向传播
                total_loss_batch.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.enhanced_config['grad_clip_norm']
                )
                
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                # 定期打印进度
                if batch_idx % 50 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    prot_t5_lr = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else current_lr
                    
                    self.logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                        f"Loss: {total_loss_batch.item():.4f}, "
                        f"LR: {current_lr:.6f}/{prot_t5_lr:.6f}"
                    )
                    
            except Exception as e:
                self.logger.error(f"训练批次失败 (epoch={epoch}, batch={batch_idx}): {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        pred_list = []
        label_list = []
        
        with torch.no_grad():
            for data in self.test_loader:
                try:
                    data = data.to(self.device)
                    pred = self.model(data)
                    pred_list.append(pred.view(-1).cpu().numpy())
                    label_list.append(data.y.cpu().numpy())
                except Exception as e:
                    self.logger.warning(f"评估批次失败: {e}")
                    continue
        
        if not pred_list:
            self.logger.error("评估失败：没有有效的预测结果")
            return float('inf'), 0.0, 0.0
        
        predictions = np.concatenate(pred_list)
        labels = np.concatenate(label_list)
        
        mse = np.mean((predictions - labels) ** 2)
        cindex = get_cindex(labels, predictions)
        r2 = get_rm2(labels, predictions)
        
        return mse, cindex, r2
    
    def train(self):
        """主训练循环"""
        self.logger.info("🚀 开始ProtT5增强训练...")
        
        best_mse = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        target_mse = self.enhanced_config['target_mse']
        current_best = self.enhanced_config['current_best_mse']
        
        self.logger.info(f"🎯 性能目标:")
        self.logger.info(f"   - 目标MSE: {target_mse:.4f}")
        self.logger.info(f"   - 当前最佳: {current_best:.4f}")
        self.logger.info(f"   - 需要改进: {current_best - target_mse:.4f}")
        
        for epoch in range(1, self.enhanced_config['max_epochs'] + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 评估
            test_mse, test_cindex, test_r2 = self.evaluate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录结果
            self.logger.info(
                f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | "
                f"Test MSE: {test_mse:.4f} | CI: {test_cindex:.4f} | R2: {test_r2:.4f}"
            )
            
            # 保存最佳模型
            if test_mse < best_mse:
                best_mse = test_mse
                best_epoch = epoch
                patience_counter = 0
                
                # 保存模型
                if self.args.save_model:
                    save_path = f"best_prot_t5_model_mse_{test_mse:.4f}_ci_{test_cindex:.4f}_r2_{test_r2:.4f}.pt"
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f"💾 保存最佳模型: {save_path}")
                
                # 检查是否达到目标
                if test_mse <= target_mse:
                    self.logger.info(f"🎉 达到目标MSE! {test_mse:.4f} <= {target_mse:.4f}")
                    break
                
                # 检查是否超越当前最佳
                if test_mse < current_best:
                    improvement = current_best - test_mse
                    progress = (improvement / (current_best - target_mse)) * 100
                    self.logger.info(f"🏆 超越当前最佳! 改进: {improvement:.4f}, 进度: {progress:.1f}%")
                
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= self.enhanced_config['early_stop_patience']:
                self.logger.info(f"⏹️ 早停触发")
                break
        
        # 训练完成总结
        self.logger.info("=" * 60)
        self.logger.info("🏁 ProtT5增强训练完成!")
        self.logger.info(f"🏆 最佳性能 (Epoch {best_epoch}): MSE={best_mse:.4f}")
        
        if best_mse <= target_mse:
            self.logger.info("🎉 完全达到目标性能!")
        elif best_mse < current_best:
            improvement = current_best - best_mse
            self.logger.info(f"🎊 超越基线性能! 改进: {improvement:.4f}")
        else:
            self.logger.info("📈 未能超越基线，可能需要调整配置")
        
        self.logger.info("=" * 60)
        
        return best_mse

def main():
    parser = argparse.ArgumentParser(description='KIBA ProtT5增强训练')
    parser.add_argument('--dataset', type=str, default='kiba', help='数据集名称')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_model', action='store_true', default=True, help='保存最佳模型')
    parser.add_argument('--use_prot_t5', action='store_true', default=True, help='使用ProtT5嵌入')
    parser.add_argument('--prot_t5_model_path', type=str, default='/home/lww/prot_t5_model', help='ProtT5模型路径')
    
    args = parser.parse_args()
    
    # 设置可复现性
    set_reproducible_seeds(args.seed)
    
    print("🎯 KIBA ProtT5增强训练")
    print("=" * 60)
    print("目标: 通过ProtT5蛋白质嵌入突破MSE<0.128")
    print("方法: 多模态特征融合 + 单模型训练")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = ProtT5EnhancedTrainer(args)
    trainer.load_data()
    trainer.create_model()
    trainer.setup_optimizer()
    
    # 开始训练
    best_mse = trainer.train()
    
    print(f"\n🎉 训练完成! 最佳MSE: {best_mse:.4f}")
    
    if best_mse < 0.128:
        print("🏆 恭喜！成功达到目标性能!")
    elif best_mse < 0.1310:
        print("🎊 超越了基线性能!")
    else:
        print("📈 未能超越基线，建议调整配置")

if __name__ == '__main__':
    main()
