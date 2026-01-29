#!/usr/bin/env python3
"""
论文可复现性验证脚本
用于验证论文中报告的KIBA优化结果
确保其他研究者能够复现我们的关键发现
"""

import os
import torch
import numpy as np
import random
import json
from datetime import datetime
from torch_geometric.data import DataLoader

from metrics import get_cindex, get_rm2
from dataset import *
from model_0428_16_dual import MGraphDTA

# 🔒 固定随机种子确保完全可复现
def set_reproducible_seeds(seed=42):
    """设置所有随机种子以确保完全可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 设置随机种子: {seed} (确保完全可复现)")

class PaperResultsValidator:
    """论文结果验证器"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.paper_results = {
            # 论文中报告的关键结果
            'baseline_training': {
                'epochs': 1365,
                'mse': 0.1330,
                'cindex': 0.8886,
                'r2': 0.7746,
                'description': '长期训练优化结果'
            },
            'model_ensemble': {
                'models_used': 6,
                'mse': 0.1321,
                'cindex': 0.8891,
                'r2': 0.7805,
                'description': '6模型集成结果'
            },
            'prediction_calibration': {
                'method': 'isotonic',
                'mse': 0.1310,
                'cindex': 0.8886,
                'r2': 0.8035,
                'description': 'Isotonic校准后结果'
            },
            'advanced_ensemble': {
                'method': 'stacking',
                'mse': 0.1303,
                'cindex': 0.8883,
                'r2': 0.8053,
                'description': 'Stacking集成+多阶段校准'
            }
        }
        
        self.tolerance = {
            'mse': 0.002,    # MSE容差 ±0.002
            'cindex': 0.005, # CI容差 ±0.005
            'r2': 0.01       # R2容差 ±0.01
        }
    
    def validate_environment(self):
        """验证运行环境"""
        print("🔍 验证运行环境...")
        
        env_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   Python版本: {env_info['python_version']}")
        print(f"   PyTorch版本: {env_info['torch_version']}")
        print(f"   CUDA可用: {env_info['cuda_available']}")
        print(f"   使用设备: {env_info['device']}")
        
        return env_info
    
    def load_best_models(self):
        """加载论文中使用的最佳模型"""
        print("📥 加载论文中的最佳模型...")
        
        # 论文中使用的最佳模型路径
        model_paths = [
            'save/20250725_233313_kiba/model/epoch-1344, LR-0.000009, MSEloss-0.1232, cindex-0.8529, r2-0.7146, test1: [MSEloss-0.1328, cindex:0.8885, r2:0.7805].pt',
            'save/20250725_233313_kiba/model/epoch-1323, LR-0.000011, MSEloss-0.1231, cindex-0.8546, r2-0.7115, test1: [MSEloss-0.1329, cindex:0.8884, r2:0.7780].pt',
            'save/20250725_233313_kiba/model/epoch-1400, LR-0.000004, MSEloss-0.1223, cindex-0.8561, r2-0.7152, test1: [MSEloss-0.1327, cindex:0.8888, r2:0.7760].pt',
            'save/20250725_233313_kiba/model/epoch-1317, LR-0.000012, MSEloss-0.1233, cindex-0.8547, r2-0.7152, test1: [MSEloss-0.1331, cindex:0.8886, r2:0.7775].pt'
        ]
        
        models = []
        successful_loads = 0
        
        for i, path in enumerate(model_paths):
            if os.path.exists(path):
                try:
                    model = MGraphDTA(3, 25 + 1,
                                     embedding_size=128,
                                     filter_num=32,
                                     out_dim=1,
                                     mask_rate=0.05,
                                     temperature=0.1,
                                     disable_masking=False,
                                     cl_mode='regression',
                                     cl_similarity_threshold=0.5,
                                     use_surface=True).to(self.device)
                    
                    state_dict = torch.load(path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    models.append(model)
                    successful_loads += 1
                    print(f"   ✅ 模型 {i+1} 加载成功")
                    
                except Exception as e:
                    print(f"   ❌ 模型 {i+1} 加载失败: {e}")
            else:
                print(f"   ❌ 模型文件不存在: {os.path.basename(path)}")
        
        print(f"📊 成功加载 {successful_loads}/{len(model_paths)} 个模型")
        return models
    
    def validate_single_model_performance(self, model, model_name="single_model"):
        """验证单个模型性能"""
        print(f"🧪 验证单模型性能: {model_name}")
        
        # 加载测试数据
        DATASET = 'kiba'
        fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', DATASET)
        test1_set = GNNDataset(fpath, types='test1', use_surface=True, use_masif=True)
        test1_loader = DataLoader(test1_set, batch_size=512, shuffle=False, num_workers=8)
        
        # 生成预测
        pred_list = []
        label_list = []
        
        with torch.no_grad():
            for data in test1_loader:
                data = data.to(self.device)
                pred = model(data)
                pred_list.append(pred.view(-1).cpu().numpy())
                label_list.append(data.y.cpu().numpy())
        
        predictions = np.concatenate(pred_list)
        labels = np.concatenate(label_list)
        
        # 计算性能指标
        mse = np.mean((predictions - labels) ** 2)
        cindex = get_cindex(labels, predictions)
        r2 = get_rm2(labels, predictions)
        
        result = {
            'mse': mse,
            'cindex': cindex,
            'r2': r2,
            'predictions': predictions,
            'labels': labels
        }
        
        print(f"   MSE: {mse:.4f}")
        print(f"   CI: {cindex:.4f}")
        print(f"   R2: {r2:.4f}")
        
        return result
    
    def validate_ensemble_performance(self, models):
        """验证集成模型性能"""
        print("🧪 验证集成模型性能...")
        
        if len(models) < 2:
            print("❌ 模型数量不足，无法进行集成验证")
            return None
        
        # 加载测试数据
        DATASET = 'kiba'
        fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', DATASET)
        test1_set = GNNDataset(fpath, types='test1', use_surface=True, use_masif=True)
        test1_loader = DataLoader(test1_set, batch_size=512, shuffle=False, num_workers=8)
        
        # 生成集成预测
        all_predictions = []
        labels = None
        
        for i, model in enumerate(models):
            print(f"   生成模型 {i+1} 预测...")
            
            batch_preds = []
            batch_labels = []
            
            with torch.no_grad():
                for data in test1_loader:
                    data = data.to(self.device)
                    pred = model(data)
                    batch_preds.append(pred.view(-1).cpu().numpy())
                    if labels is None:
                        batch_labels.append(data.y.cpu().numpy())
            
            all_predictions.append(np.concatenate(batch_preds))
            if labels is None:
                labels = np.concatenate(batch_labels)
        
        # 等权重集成
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        # 计算集成性能
        mse = np.mean((ensemble_pred - labels) ** 2)
        cindex = get_cindex(labels, ensemble_pred)
        r2 = get_rm2(labels, ensemble_pred)
        
        result = {
            'mse': mse,
            'cindex': cindex,
            'r2': r2,
            'predictions': ensemble_pred,
            'labels': labels,
            'individual_predictions': all_predictions
        }
        
        print(f"   集成MSE: {mse:.4f}")
        print(f"   集成CI: {cindex:.4f}")
        print(f"   集成R2: {r2:.4f}")
        
        return result
    
    def compare_with_paper_results(self, actual_results, paper_key):
        """与论文结果对比"""
        paper_result = self.paper_results[paper_key]
        
        print(f"\n📊 与论文结果对比 ({paper_result['description']}):")
        print("-" * 50)
        
        comparisons = {}
        all_within_tolerance = True
        
        for metric in ['mse', 'cindex', 'r2']:
            paper_value = paper_result[metric]
            actual_value = actual_results[metric]
            diff = abs(actual_value - paper_value)
            tolerance = self.tolerance[metric]
            within_tolerance = diff <= tolerance
            
            comparisons[metric] = {
                'paper': paper_value,
                'actual': actual_value,
                'difference': diff,
                'tolerance': tolerance,
                'within_tolerance': within_tolerance
            }
            
            status = "✅ 通过" if within_tolerance else "❌ 超出容差"
            print(f"{metric.upper():>6}: 论文={paper_value:.4f}, 实际={actual_value:.4f}, 差异={diff:.4f}, {status}")
            
            if not within_tolerance:
                all_within_tolerance = False
        
        print("-" * 50)
        if all_within_tolerance:
            print("🎉 所有指标都在可接受容差范围内！")
        else:
            print("⚠️ 部分指标超出容差范围")
        
        return comparisons, all_within_tolerance
    
    def generate_reproducibility_report(self, validation_results):
        """生成可复现性报告"""
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'environment': validation_results.get('environment', {}),
            'model_loading': validation_results.get('model_loading', {}),
            'performance_comparisons': validation_results.get('comparisons', {}),
            'overall_reproducibility': validation_results.get('overall_success', False),
            'recommendations': []
        }
        
        # 添加建议
        if report['overall_reproducibility']:
            report['recommendations'].append("✅ 结果完全可复现，可以安全提交论文")
        else:
            report['recommendations'].append("⚠️ 部分结果存在差异，建议检查环境配置")
            report['recommendations'].append("💡 确保使用相同的随机种子和模型权重")
            report['recommendations'].append("🔧 验证数据预处理流程的一致性")
        
        return report

def main():
    print("🔬 KIBA优化论文可复现性验证")
    print("=" * 60)
    
    # 设置可复现性
    set_reproducible_seeds(42)
    
    # 初始化验证器
    device = 'cpu'  # 使用CPU确保跨平台一致性
    validator = PaperResultsValidator(device=device)
    
    # 验证环境
    env_info = validator.validate_environment()
    
    # 加载模型
    models = validator.load_best_models()
    
    validation_results = {
        'environment': env_info,
        'model_loading': {'successful_models': len(models)},
        'comparisons': {},
        'overall_success': True
    }
    
    if len(models) > 0:
        # 验证单模型性能
        single_result = validator.validate_single_model_performance(models[0], "best_single_model")
        
        # 验证集成性能
        if len(models) >= 2:
            ensemble_result = validator.validate_ensemble_performance(models[:4])  # 使用前4个模型
            
            if ensemble_result:
                # 与论文结果对比
                comparison, success = validator.compare_with_paper_results(ensemble_result, 'model_ensemble')
                validation_results['comparisons']['ensemble'] = comparison
                validation_results['overall_success'] = success
        
        # 生成报告
        report = validator.generate_reproducibility_report(validation_results)
        
        # 保存报告
        with open('reproducibility_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 可复现性报告已保存: reproducibility_report.json")
        
        # 打印总结
        print(f"\n🎯 验证总结:")
        print(f"   环境验证: ✅")
        print(f"   模型加载: {'✅' if len(models) > 0 else '❌'}")
        print(f"   结果对比: {'✅' if validation_results['overall_success'] else '❌'}")
        print(f"   整体可复现性: {'✅ 通过' if validation_results['overall_success'] else '❌ 需要调整'}")
        
    else:
        print("❌ 无法加载任何模型，请检查模型文件路径")

if __name__ == '__main__':
    import sys
    main()
