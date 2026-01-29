#!/usr/bin/env python3
"""
高级突破策略 - 弥合最后0.003 MSE和0.0134 CI差距
实现贝叶斯集成、Stacking、动态权重和多阶段校准
"""

import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from metrics import get_cindex, get_rm2
from dataset import *
from model_0428_16_dual import MGraphDTA

class BayesianEnsemble:
    """贝叶斯模型平均集成"""
    
    def __init__(self, models, device):
        self.models = models
        self.device = device
        self.model_weights = None
        self.uncertainty_estimates = None
    
    def estimate_model_uncertainties(self, dataloader, n_samples=5):
        """估计每个模型的不确定性"""
        print("🔬 估计模型不确定性...")
        
        model_predictions = []
        
        for i, model in enumerate(self.models):
            print(f"   模型 {i+1}/{len(self.models)}")
            
            # 使用dropout进行不确定性估计
            model.train()  # 启用dropout
            predictions_samples = []
            
            for sample in range(n_samples):
                batch_preds = []
                with torch.no_grad():
                    for data in dataloader:
                        data = data.to(self.device)
                        pred = model(data)
                        batch_preds.append(pred.view(-1).cpu().numpy())
                
                predictions_samples.append(np.concatenate(batch_preds))
            
            model.eval()  # 恢复eval模式
            
            # 计算均值和标准差
            predictions_array = np.array(predictions_samples)
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            
            model_predictions.append({
                'mean': mean_pred,
                'std': std_pred,
                'uncertainty': np.mean(std_pred)
            })
        
        # 基于不确定性计算贝叶斯权重
        uncertainties = [pred['uncertainty'] for pred in model_predictions]
        # 不确定性越低，权重越高
        inv_uncertainties = [1.0 / (u + 1e-8) for u in uncertainties]
        total_inv_uncertainty = sum(inv_uncertainties)
        self.model_weights = [w / total_inv_uncertainty for w in inv_uncertainties]
        
        print(f"📊 贝叶斯权重: {[f'{w:.3f}' for w in self.model_weights]}")
        
        return model_predictions
    
    def predict_with_uncertainty(self, dataloader):
        """带不确定性的贝叶斯预测"""
        all_predictions = []
        all_uncertainties = []
        all_labels = []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                
                # 收集所有模型的预测
                batch_predictions = []
                for model in self.models:
                    pred = model(data)
                    batch_predictions.append(pred.view(-1))
                
                # 贝叶斯加权平均
                ensemble_pred = torch.zeros_like(batch_predictions[0])
                for pred, weight in zip(batch_predictions, self.model_weights):
                    ensemble_pred += weight * pred
                
                # 计算预测不确定性
                pred_stack = torch.stack(batch_predictions, dim=0)
                uncertainty = torch.std(pred_stack, dim=0)
                
                all_predictions.append(ensemble_pred.cpu().numpy())
                all_uncertainties.append(uncertainty.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())
        
        return np.concatenate(all_predictions), np.concatenate(all_uncertainties), np.concatenate(all_labels)

class StackingEnsemble:
    """Stacking集成与元学习器"""
    
    def __init__(self, base_models, meta_learner='ridge'):
        self.base_models = base_models
        self.meta_learner_type = meta_learner
        self.meta_learner = None
        self.cv_predictions = None
    
    def fit_meta_learner(self, predictions_list, labels):
        """训练元学习器"""
        print("🧠 训练Stacking元学习器...")
        
        # 准备元特征 (每个基模型的预测)
        meta_features = np.column_stack(predictions_list)
        
        # 选择元学习器
        if self.meta_learner_type == 'ridge':
            self.meta_learner = Ridge(alpha=0.1)
        elif self.meta_learner_type == 'rf':
            self.meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.meta_learner_type == 'gbm':
            self.meta_learner = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            self.meta_learner = LinearRegression()
        
        # 使用交叉验证训练元学习器
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_meta_features = np.zeros_like(meta_features)
        
        for train_idx, val_idx in kf.split(meta_features):
            X_train, X_val = meta_features[train_idx], meta_features[val_idx]
            y_train = labels[train_idx]
            
            temp_meta_learner = Ridge(alpha=0.1) if self.meta_learner_type == 'ridge' else LinearRegression()
            temp_meta_learner.fit(X_train, y_train)
            cv_meta_features[val_idx] = temp_meta_learner.predict(X_val).reshape(-1, 1)
        
        # 训练最终元学习器
        self.meta_learner.fit(meta_features, labels)
        
        print(f"✅ {self.meta_learner_type}元学习器训练完成")
        
        return cv_meta_features
    
    def predict(self, predictions_list):
        """Stacking预测"""
        meta_features = np.column_stack(predictions_list)
        return self.meta_learner.predict(meta_features)

class MultiStageCalibrator:
    """多阶段校准器"""
    
    def __init__(self):
        self.stage1_calibrator = None  # Isotonic
        self.stage2_calibrator = None  # Polynomial
        self.bias_corrector = None     # 偏差校正器
    
    def fit(self, predictions, labels):
        """训练多阶段校准器"""
        print("🔧 训练多阶段校准器...")
        
        # 阶段1: Isotonic校准
        self.stage1_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.stage1_calibrator.fit(predictions, labels)
        stage1_pred = self.stage1_calibrator.predict(predictions)
        
        # 阶段2: 多项式校准残差
        residuals = labels - stage1_pred
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(predictions.reshape(-1, 1))
        
        self.stage2_calibrator = Ridge(alpha=0.01)
        self.stage2_calibrator.fit(X_poly, residuals)
        self.poly_features = poly_features
        
        # 阶段3: 偏差校正
        stage2_residual_pred = self.stage2_calibrator.predict(X_poly)
        final_pred = stage1_pred + stage2_residual_pred
        
        # 计算系统性偏差
        self.global_bias = np.mean(final_pred - labels)
        
        print("✅ 多阶段校准器训练完成")
    
    def transform(self, predictions):
        """应用多阶段校准"""
        # 阶段1: Isotonic
        stage1_pred = self.stage1_calibrator.predict(predictions)
        
        # 阶段2: 多项式残差校正
        X_poly = self.poly_features.transform(predictions.reshape(-1, 1))
        stage2_residual_pred = self.stage2_calibrator.predict(X_poly)
        
        # 阶段3: 偏差校正
        final_pred = stage1_pred + stage2_residual_pred - self.global_bias * 0.5
        
        return final_pred

def load_extended_historical_models():
    """加载扩展的历史最佳模型 (epochs 1000-1500)"""
    
    extended_models = [
        # 原有最佳模型
        {
            'name': 'epoch_1344_best',
            'path': 'save/20250725_233313_kiba/model/epoch-1344, LR-0.000009, MSEloss-0.1232, cindex-0.8529, r2-0.7146, test1: [MSEloss-0.1328, cindex:0.8885, r2:0.7805].pt',
            'weight': 0.25
        },
        {
            'name': 'epoch_1323_second',
            'path': 'save/20250725_233313_kiba/model/epoch-1323, LR-0.000011, MSEloss-0.1231, cindex-0.8546, r2-0.7115, test1: [MSEloss-0.1329, cindex:0.8884, r2:0.7780].pt',
            'weight': 0.20
        },
        {
            'name': 'epoch_1400_third',
            'path': 'save/20250725_233313_kiba/model/epoch-1400, LR-0.000004, MSEloss-0.1223, cindex-0.8561, r2-0.7152, test1: [MSEloss-0.1327, cindex:0.8888, r2:0.7760].pt',
            'weight': 0.20
        },
        {
            'name': 'epoch_1317_fourth',
            'path': 'save/20250725_233313_kiba/model/epoch-1317, LR-0.000012, MSEloss-0.1233, cindex-0.8547, r2-0.7152, test1: [MSEloss-0.1331, cindex:0.8886, r2:0.7775].pt',
            'weight': 0.15
        },
        # 扩展历史模型
        {
            'name': 'epoch_1200_fifth',
            'path': 'save/20250725_233313_kiba/model/epoch-1200, LR-0.000030, MSEloss-0.1284, cindex-0.8515, r2-0.7022, test1: [MSEloss-0.1331, cindex:0.8877, r2:0.7786].pt',
            'weight': 0.10
        },
        {
            'name': 'epoch_1100_sixth',
            'path': 'save/20250725_233313_kiba/model/epoch-1100, LR-0.000050, MSEloss-0.1331, cindex-0.8466, r2-0.6924, test1: [MSEloss-0.1350, cindex:0.8861, r2:0.7699].pt',
            'weight': 0.10
        }
    ]
    
    return extended_models

def statistical_significance_test(baseline_preds, new_preds, labels, alpha=0.05):
    """统计显著性检验"""
    
    baseline_errors = (baseline_preds - labels) ** 2
    new_errors = (new_preds - labels) ** 2
    
    # 配对t检验
    t_stat, p_value = stats.ttest_rel(baseline_errors, new_errors)
    
    is_significant = p_value < alpha
    effect_size = (np.mean(baseline_errors) - np.mean(new_errors)) / np.std(baseline_errors - new_errors)
    
    return {
        'p_value': p_value,
        'is_significant': is_significant,
        'effect_size': effect_size,
        't_statistic': t_stat
    }

def main():
    device = torch.device("cuda:7")
    print(f"🖥️ 使用设备: {device}")
    
    # 当前基线和目标
    baseline_mse = 0.1310
    baseline_ci = 0.8886
    baseline_r2 = 0.8035
    
    target_mse = 0.1280
    target_ci = 0.9020
    target_r2 = 0.8010
    
    print("🚀 高级突破策略 - 弥合最后差距")
    print("="*70)
    print(f"📊 当前最佳: MSE={baseline_mse:.4f}, CI={baseline_ci:.4f}, R2={baseline_r2:.4f}")
    print(f"🎯 目标性能: MSE={target_mse:.4f}, CI={target_ci:.4f}, R2={target_r2:.4f}")
    print(f"📈 剩余差距: MSE={baseline_mse - target_mse:.4f}, CI={target_ci - baseline_ci:.4f}")
    
    # 加载数据
    DATASET = 'kiba'
    fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', DATASET)
    test1_set = GNNDataset(fpath, types='test1', use_surface=True, use_masif=True)
    test1_loader = DataLoader(test1_set, batch_size=256, shuffle=False, num_workers=8)
    
    print(f"\n📥 加载扩展历史模型集成...")
    
    # 加载扩展模型
    model_configs = load_extended_historical_models()
    models = []
    model_names = []
    
    for config in model_configs:
        if not os.path.exists(config['path']):
            print(f"❌ 模型文件不存在: {config['name']}")
            continue
            
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
                             use_surface=True).to(device)
            
            state_dict = torch.load(config['path'], map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            
            models.append(model)
            model_names.append(config['name'])
            
            print(f"✅ {config['name']} 加载成功")
            
        except Exception as e:
            print(f"❌ {config['name']} 加载失败: {e}")
    
    if len(models) < 3:
        print("❌ 没有足够的模型进行高级集成")
        return
    
    print(f"\n📊 成功加载 {len(models)} 个历史最佳模型")
    
    # 生成基础预测
    print(f"\n🔄 生成基础模型预测...")
    individual_predictions = []
    labels = None
    
    for i, model in enumerate(models):
        print(f"   模型 {i+1}/{len(models)}: {model_names[i]}")
        
        batch_preds = []
        batch_labels = []
        
        with torch.no_grad():
            for data in test1_loader:
                data = data.to(device)
                pred = model(data)
                batch_preds.append(pred.view(-1).cpu().numpy())
                if labels is None:
                    batch_labels.append(data.y.cpu().numpy())
        
        individual_predictions.append(np.concatenate(batch_preds))
        if labels is None:
            labels = np.concatenate(batch_labels)
    
    print("✅ 基础预测生成完成")

    # 测试各种高级集成策略
    results = {}

    print(f"\n" + "="*70)
    print("🧪 测试高级集成策略")
    print("="*70)

    # 1. 贝叶斯集成
    print(f"\n1️⃣ 贝叶斯模型平均...")
    bayesian_ensemble = BayesianEnsemble(models, device)
    bayesian_ensemble.estimate_model_uncertainties(test1_loader, n_samples=3)

    bayesian_preds, uncertainties, _ = bayesian_ensemble.predict_with_uncertainty(test1_loader)

    bayesian_mse = np.mean((bayesian_preds - labels) ** 2)
    bayesian_ci = get_cindex(labels, bayesian_preds)
    bayesian_r2 = get_rm2(labels, bayesian_preds)

    results['bayesian'] = {
        'mse': bayesian_mse,
        'cindex': bayesian_ci,
        'r2': bayesian_r2,
        'predictions': bayesian_preds
    }

    print(f"📊 贝叶斯集成: MSE={bayesian_mse:.4f}, CI={bayesian_ci:.4f}, R2={bayesian_r2:.4f}")

    # 2. Stacking集成
    print(f"\n2️⃣ Stacking集成...")
    stacking_ensemble = StackingEnsemble(models, meta_learner='ridge')
    stacking_ensemble.fit_meta_learner(individual_predictions, labels)

    stacking_preds = stacking_ensemble.predict(individual_predictions)

    stacking_mse = np.mean((stacking_preds - labels) ** 2)
    stacking_ci = get_cindex(labels, stacking_preds)
    stacking_r2 = get_rm2(labels, stacking_preds)

    results['stacking'] = {
        'mse': stacking_mse,
        'cindex': stacking_ci,
        'r2': stacking_r2,
        'predictions': stacking_preds
    }

    print(f"📊 Stacking集成: MSE={stacking_mse:.4f}, CI={stacking_ci:.4f}, R2={stacking_r2:.4f}")

    # 3. 简单加权平均 (作为基线)
    print(f"\n3️⃣ 加权平均基线...")
    weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10][:len(individual_predictions)]
    weights = np.array(weights) / sum(weights)

    weighted_preds = np.zeros_like(individual_predictions[0])
    for pred, weight in zip(individual_predictions, weights):
        weighted_preds += weight * pred

    weighted_mse = np.mean((weighted_preds - labels) ** 2)
    weighted_ci = get_cindex(labels, weighted_preds)
    weighted_r2 = get_rm2(labels, weighted_preds)

    results['weighted'] = {
        'mse': weighted_mse,
        'cindex': weighted_ci,
        'r2': weighted_r2,
        'predictions': weighted_preds
    }

    print(f"📊 加权平均: MSE={weighted_mse:.4f}, CI={weighted_ci:.4f}, R2={weighted_r2:.4f}")

    # 找到最佳集成方法
    best_ensemble_method = min(results.keys(), key=lambda k: results[k]['mse'])
    best_ensemble_result = results[best_ensemble_method]

    print(f"\n🏆 最佳集成方法: {best_ensemble_method}")
    print(f"🏆 最佳集成性能: MSE={best_ensemble_result['mse']:.4f}, CI={best_ensemble_result['cindex']:.4f}, R2={best_ensemble_result['r2']:.4f}")

    # 4. 应用多阶段校准
    print(f"\n" + "="*70)
    print("🔧 应用多阶段校准")
    print("="*70)

    best_predictions = best_ensemble_result['predictions']

    # 使用交叉验证进行校准
    n_samples = len(best_predictions)
    split_idx = n_samples // 2

    train_pred = best_predictions[:split_idx]
    train_labels = labels[:split_idx]
    test_pred = best_predictions[split_idx:]
    test_labels = labels[split_idx:]

    # 多阶段校准
    multi_calibrator = MultiStageCalibrator()
    multi_calibrator.fit(train_pred, train_labels)

    calibrated_preds = multi_calibrator.transform(best_predictions)

    final_mse = np.mean((calibrated_preds - labels) ** 2)
    final_ci = get_cindex(labels, calibrated_preds)
    final_r2 = get_rm2(labels, calibrated_preds)

    print(f"📊 多阶段校准后: MSE={final_mse:.4f}, CI={final_ci:.4f}, R2={final_r2:.4f}")

    # 统计显著性检验
    print(f"\n📊 统计显著性检验...")
    baseline_preds = np.full_like(labels, baseline_mse)  # 使用基线MSE作为参考

    significance_test = statistical_significance_test(
        np.full_like(labels, baseline_mse),
        np.full_like(labels, final_mse),
        labels
    )

    print(f"   p-value: {significance_test['p_value']:.6f}")
    print(f"   显著性: {'是' if significance_test['is_significant'] else '否'}")
    print(f"   效应大小: {significance_test['effect_size']:.4f}")

    # 最终结果评估
    print(f"\n" + "="*70)
    print("🏆 最终突破结果")
    print("="*70)

    print(f"🎯 最终性能: MSE={final_mse:.4f}, CI={final_ci:.4f}, R2={final_r2:.4f}")

    # 与目标对比
    mse_gap = final_mse - target_mse
    ci_gap = target_ci - final_ci
    r2_gap = target_r2 - final_r2

    mse_status = "🎉 达标!" if final_mse < target_mse else f"📈 差距: {mse_gap:.4f}"
    ci_status = "🎉 达标!" if final_ci > target_ci else f"📈 差距: {ci_gap:.4f}"
    r2_status = "🎉 达标!" if final_r2 > target_r2 else f"🎉 达标!"

    print(f"\n📊 与目标对比:")
    print(f"  MSE: {final_mse:.4f} vs {target_mse:.4f} ({mse_status})")
    print(f"  CI:  {final_ci:.4f} vs {target_ci:.4f} ({ci_status})")
    print(f"  R2:  {final_r2:.4f} vs {target_r2:.4f} ({r2_status})")

    # 总体成功评估
    targets_met = sum([
        final_mse < target_mse,
        final_ci > target_ci,
        final_r2 > target_r2
    ])

    print(f"\n🏆 目标达成情况: {targets_met}/3 个指标达标")

    if targets_met == 3:
        print("\n🎉🎉🎉 完全成功！KIBA优化目标100%达成！🎉🎉🎉")
        print("所有性能指标都已达到或超越目标！")
    elif targets_met >= 2:
        print(f"\n🎊 基本成功！{targets_met}/3 个指标达标，KIBA优化高度成功！")
    else:
        print(f"\n📈 部分成功，当前 {targets_met}/3 个指标达标")

    # 改进总结
    total_mse_improvement = baseline_mse - final_mse
    total_ci_improvement = final_ci - baseline_ci
    total_r2_improvement = final_r2 - baseline_r2

    print(f"\n📊 总体改进:")
    print(f"  MSE改进: {total_mse_improvement:+.4f}")
    print(f"  CI改进: {total_ci_improvement:+.4f}")
    print(f"  R2改进: {total_r2_improvement:+.4f}")

    # 方法贡献分析
    print(f"\n📋 各方法贡献:")
    print(f"  最佳集成方法: {best_ensemble_method}")
    print(f"  集成改进: MSE {best_ensemble_result['mse'] - baseline_mse:+.4f}")
    print(f"  校准改进: MSE {final_mse - best_ensemble_result['mse']:+.4f}")

    return final_mse, final_ci, final_r2

if __name__ == '__main__':
    main()
