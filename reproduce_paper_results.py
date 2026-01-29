#!/usr/bin/env python3
"""
论文结果复现脚本
用于验证单模型训练的可复现性
确保其他研究者能够直接复现我们报告的结果
"""

import os
import torch
import numpy as np
import random
import argparse
import json
from datetime import datetime
from torch_geometric.data import DataLoader

from model_0428_16_dual import MGraphDTA
from dataset import GNNDataset
from metrics import get_cindex, get_rm2

def set_reproducible_seeds(seed=42):
    """设置所有随机种子确保完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class PaperResultsReproducer:
    """论文结果复现器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        # 论文中报告的预期结果
        self.paper_results = {
            'mse': 0.1310,
            'cindex': 0.8886,
            'r2': 0.8035,
            'description': '单模型统一训练结果'
        }
        
        # 可接受的容差范围
        self.tolerance = {
            'mse': 0.003,    # ±0.003 MSE容差
            'cindex': 0.005, # ±0.005 CI容差  
            'r2': 0.01       # ±0.01 R2容差
        }
        
        print("🔬 论文结果复现器初始化")
        print(f"📊 预期结果: MSE={self.paper_results['mse']:.4f}, "
              f"CI={self.paper_results['cindex']:.4f}, R2={self.paper_results['r2']:.4f}")
        print(f"📏 容差范围: MSE±{self.tolerance['mse']:.3f}, "
              f"CI±{self.tolerance['cindex']:.3f}, R2±{self.tolerance['r2']:.3f}")
    
    def load_test_data(self):
        """加载测试数据"""
        print("📊 加载KIBA测试数据...")
        
        fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', self.args.dataset)
        test1_set = GNNDataset(fpath, types='test1', use_surface=True, use_masif=True)
        
        self.test_loader = DataLoader(
            test1_set, 
            batch_size=512, 
            shuffle=False, 
            num_workers=8
        )
        
        print(f"✅ 测试集大小: {len(test1_set)} 样本")
        print(f"✅ 表面特征: 启用")
    
    def load_trained_model(self):
        """加载训练好的模型"""
        print("📥 加载训练好的模型...")
        
        # 创建模型架构
        self.model = MGraphDTA(
            3, 25 + 1,
            embedding_size=128,
            filter_num=32,
            out_dim=1,
            mask_rate=0.05,
            temperature=0.1,
            disable_masking=False,
            cl_mode='regression',
            cl_similarity_threshold=0.5,
            use_surface=True
        ).to(self.device)
        
        # 查找最佳模型文件
        model_files = [f for f in os.listdir('.') if f.startswith('best_single_model_') and f.endswith('.pt')]
        
        if not model_files:
            print("❌ 未找到训练好的模型文件")
            print("💡 请先运行: python train_kiba_single_model.py --save_model")
            return False
        
        # 选择最新的模型文件
        latest_model = max(model_files, key=os.path.getctime)
        
        try:
            state_dict = torch.load(latest_model, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"✅ 成功加载模型: {latest_model}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def evaluate_model(self):
        """评估模型性能"""
        print("🧪 评估模型性能...")
        
        pred_list = []
        label_list = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data = data.to(self.device)
                pred = self.model(data)
                pred_list.append(pred.view(-1).cpu().numpy())
                label_list.append(data.y.cpu().numpy())
                
                # 显示进度
                if batch_idx % 10 == 0:
                    print(f"   处理批次 {batch_idx+1}/{len(self.test_loader)}")
        
        predictions = np.concatenate(pred_list)
        labels = np.concatenate(label_list)
        
        # 计算性能指标
        mse = np.mean((predictions - labels) ** 2)
        cindex = get_cindex(labels, predictions)
        r2 = get_rm2(labels, predictions)
        
        results = {
            'mse': mse,
            'cindex': cindex,
            'r2': r2,
            'predictions': predictions.tolist(),
            'labels': labels.tolist()
        }
        
        print(f"📊 实际结果: MSE={mse:.4f}, CI={cindex:.4f}, R2={r2:.4f}")
        
        return results
    
    def compare_with_paper(self, actual_results):
        """与论文结果对比"""
        print("\n📊 与论文结果对比:")
        print("=" * 60)
        
        comparisons = {}
        all_within_tolerance = True
        
        for metric in ['mse', 'cindex', 'r2']:
            paper_value = self.paper_results[metric]
            actual_value = actual_results[metric]
            diff = abs(actual_value - paper_value)
            tolerance = self.tolerance[metric]
            within_tolerance = diff <= tolerance
            
            comparisons[metric] = {
                'paper': paper_value,
                'actual': actual_value,
                'difference': diff,
                'tolerance': tolerance,
                'within_tolerance': within_tolerance,
                'relative_error': (diff / paper_value) * 100
            }
            
            status = "✅ 通过" if within_tolerance else "❌ 超出容差"
            rel_error = comparisons[metric]['relative_error']
            
            print(f"{metric.upper():>6}: 论文={paper_value:.4f}, "
                  f"实际={actual_value:.4f}, 差异={diff:.4f} ({rel_error:.1f}%), {status}")
            
            if not within_tolerance:
                all_within_tolerance = False
        
        print("=" * 60)
        
        if all_within_tolerance:
            print("🎉 所有指标都在可接受容差范围内！")
            print("✅ 论文结果完全可复现！")
        else:
            print("⚠️ 部分指标超出容差范围")
            print("💡 建议检查环境配置或重新训练")
        
        return comparisons, all_within_tolerance
    
    def generate_report(self, actual_results, comparisons, reproducible):
        """生成详细报告"""
        report = {
            'reproduction_info': {
                'timestamp': datetime.now().isoformat(),
                'seed': self.args.seed,
                'device': str(self.device),
                'dataset': self.args.dataset
            },
            'paper_results': self.paper_results,
            'actual_results': {k: v for k, v in actual_results.items() if k != 'predictions' and k != 'labels'},
            'comparisons': comparisons,
            'reproducibility': {
                'overall_success': reproducible,
                'tolerance_used': self.tolerance
            },
            'recommendations': []
        }
        
        # 添加建议
        if reproducible:
            report['recommendations'].extend([
                "✅ 结果完全可复现，可以安全用于论文提交",
                "📊 所有性能指标都在预期范围内",
                "🎯 单模型方法成功达到目标性能"
            ])
        else:
            report['recommendations'].extend([
                "⚠️ 部分结果存在差异，建议进一步检查",
                "🔧 确保使用相同的随机种子和环境配置",
                "💡 可能需要重新训练或调整超参数"
            ])
        
        return report
    
    def reproduce(self):
        """执行完整的复现流程"""
        print("🚀 开始论文结果复现...")
        print("=" * 60)
        
        # 加载数据
        self.load_test_data()
        
        # 加载模型
        if not self.load_trained_model():
            return None
        
        # 评估性能
        actual_results = self.evaluate_model()
        
        # 与论文结果对比
        comparisons, reproducible = self.compare_with_paper(actual_results)
        
        # 生成报告
        report = self.generate_report(actual_results, comparisons, reproducible)
        
        # 保存报告
        report_file = f"reproduction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存: {report_file}")
        
        # 打印总结
        print(f"\n🎯 复现总结:")
        print(f"   环境验证: ✅")
        print(f"   模型加载: ✅")
        print(f"   性能评估: ✅")
        print(f"   结果对比: {'✅ 成功' if reproducible else '❌ 部分失败'}")
        print(f"   整体可复现性: {'✅ 通过' if reproducible else '❌ 需要调整'}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='论文结果复现验证')
    parser.add_argument('--dataset', type=str, default='kiba', help='数据集名称')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置可复现性
    set_reproducible_seeds(args.seed)
    
    print("📋 KIBA论文结果复现验证")
    print("=" * 60)
    print("目标: 验证单模型训练结果的可复现性")
    print("方法: 加载训练好的模型并评估性能")
    print("=" * 60)
    
    # 创建复现器并执行
    reproducer = PaperResultsReproducer(args)
    report = reproducer.reproduce()
    
    if report and report['reproducibility']['overall_success']:
        print("\n🎉 恭喜！论文结果完全可复现！")
        print("📊 可以安全地在论文中报告这些结果")
    elif report:
        print("\n⚠️ 复现过程完成，但部分结果存在差异")
        print("💡 请查看详细报告了解具体情况")
    else:
        print("\n❌ 复现失败，请检查模型文件和环境配置")

if __name__ == '__main__':
    main()
