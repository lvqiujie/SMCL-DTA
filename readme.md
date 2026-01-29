# KIBA优化论文提交文件清单

## 📋 核心可复现文件

### 🎯 主要结果复现脚本
```bash
paper_standard_reproduction.py     # 🏆 论文标准复现脚本 (推荐使用)
paper_reproducibility_validation.py # 🔍 可复现性验证脚本
```
❄️ 泛化能力验证脚本 (Cold-Start)
用于复现论文 Figure 1E 中的冷启动实验结果。
```bash
train_kiba_cold.py # 🥶 单边冷启动训练脚本 (Drug/Target Cold, Test1)
train_with_prot_t5.py # 🥶 双盲冷启动训练脚本 (Pair Cold, Test2, 含ProtT5)
split_cold.ipynb # 🧩 骨架与聚类划分生成器
generate_kiba_prot_t5_embeddings.py # 🧬 ProtT5 特征生成器
```
### 🔬 训练和优化脚本
```bash
train_kiba_optimized.py            # 基础训练脚本 (1365 epochs)
final_breakthrough_simple.py       # 集成+校准脚本 (最佳结果)
advanced_breakthrough_final.py     # 高级集成策略
```

### 🏗️ 核心架构文件
```bash
src/
├── model_0428_16_dual.py # 基础 MGraphDTA 模型架构 (Warm-Start)
├── model_with_prot_t5.py # ProtT5 增强模型架构 (Test2 Cold-Start)
├── dataset.py # 基础数据加载器
├── dataset_with_prot_t5.py # 增强数据加载器 (ProtT5 支持)
└── metrics.py # 评估指标
```






## 🔄 复现步骤

### Step 1: 环境配置
```bash
# 创建conda环境
conda create -n kiba_reproduction python=3.10
conda activate kiba_reproduction

# 安装依赖
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn numpy pandas
pip install rdkit-pypi
```

### Step 2: 数据准备
```bash
# 确保KIBA数据集位于正确路径
原始数据: /data/kiba
冷启动预处理数据: data/kiba/cold/processed/
ProtT5特征 (Test2必需): data/kiba/saved_protein_data/
```

### Step 3: 运行标准复现
```bash
# 运行论文标准复现脚本
python paper_standard_reproduction.py

```
```bash
# 运行 Cold-Start 泛化实验
场景 A: 复现单边冷启动 (Drug Cold)
# 加载 test1 数据集
python train_kiba_cold.py

场景 B: 复现双盲冷启动 (Pair Cold - 最难)
# 加载 test2 数据集并启用 ProtT5 增强
python train_with_prot_t5.py

```

### Step 4: 验证可复现性
```bash
# 运行可复现性验证
python paper_reproducibility_validation.py

# 检查生成的报告
cat reproducibility_report.json
```

## ✅ 可复现性保证

### 🔒 随机种子控制
```python
# 所有脚本都使用固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```



## 🚨 重要注意事项
### ✅ 必须包含的文件 (Must Include Files)
#### 1. 核心复现脚本 (Core Scripts)
* **`paper_standard_reproduction.py`**: [Warm-Start] 论文标准复现脚本 (SOTA MSE=0.1310)。
* **`train_kiba_cold.py`**: [Cold-Start] 冷启动实验主程序单边 (Test1) 。
* * **`train_with_prot_t5.py`**: [Cold-Start] 双盲 (Test2) 模式。

#### 2. 冷启动关键组件 (Cold-Start Components)
* **`split_cold.ipynb`**: **[核心划分逻辑]** 定义了药物骨架 (Scaffold) 和蛋白质聚类 (Cluster) 的划分算法，是复现 Figure 1E 数据分布的源头。
* **`generate_kiba_prot_t5_embeddings.py`**: **[特征生成]** 用于生成 ProtT5 蛋白质嵌入，是双盲冷启动 (Pair Cold) 的必要前置工具。
* **`src/model_with_prot_t5.py`**: **[增强模型]** 集成了 Attention Fusion 模块的增强版架构，专门用于处理未见过的靶点。
* **`src/dataset_with_prot_t5.py`**: **[增强加载器]** 支持加载 ProtT5 向量的数据处理类。

#### 3. 基础架构文件 (Foundation)
* **`src/model_0428_16_dual.py`**: [Warm-Start] 
* **`src/dataset.py`**: 基础数据加载器。
* **`src/metrics.py`**: 统一评估指标 (MSE, CI, R2)。

#### 4. 预训练模型权重 (Checkpoints)
**预训练模型权重** - 4个最佳模型文件

### ⚠️ 环境要求
- Python 3.10+
- PyTorch 1.12+
- torch-geometric
- scikit-learn
- 固定随机种子


**📋 总结**: 我们提供了完整的可复现性保证，包括标准化脚本、预训练模型和详细文档。其他研究者可以轻松复现我们的98%+目标达成度结果。
