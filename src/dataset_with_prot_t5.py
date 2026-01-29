#!/usr/bin/env python3
"""
增强的数据集加载器 - 集成ProtT5蛋白质嵌入
在原有特征基础上添加ProtT5预训练蛋白质表示
"""

import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch_geometric.data import Data, Dataset
from transformers import T5EncoderModel, T5Tokenizer
import warnings
warnings.filterwarnings('ignore')

# 导入原始数据集类
from dataset import GNNDataset

class ProtT5EmbeddingGenerator:
    """ProtT5嵌入生成器"""
    
    def __init__(self, model_path="/home/lww/prot_t5_model", device='cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.embedding_cache = {}
        
    def load_model(self):
        """加载ProtT5模型"""
        try:
            print(f"🔄 加载ProtT5模型从: {self.model_path}")
            self.model = T5EncoderModel.from_pretrained(self.model_path, local_files_only=True)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path, do_lower_case=False, local_files_only=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✅ ProtT5模型加载成功")
            return True
        except Exception as e:
            print(f"❌ ProtT5模型加载失败: {e}")
            print("💡 将使用预计算的嵌入或随机初始化")
            return False
    
    def get_embedding(self, sequence):
        """获取单个蛋白质序列的ProtT5嵌入"""
        if sequence in self.embedding_cache:
            return self.embedding_cache[sequence]
        
        if self.model is None:
            # 如果模型未加载，返回随机嵌入（用于测试）
            embedding = np.random.normal(0, 0.1, 1024).astype(np.float32)
            self.embedding_cache[sequence] = embedding
            return embedding
        
        try:
            # 处理特殊氨基酸
            sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            
            # ProtT5需要氨基酸之间加空格
            spaced_sequence = ' '.join(list(sequence))
            
            # 编码序列
            inputs = self.tokenizer(spaced_sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS] token的嵌入或平均池化
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # 缓存结果
            self.embedding_cache[sequence] = embedding.astype(np.float32)
            return embedding
            
        except Exception as e:
            print(f"⚠️ 序列嵌入生成失败: {e}")
            # 返回零向量作为fallback
            embedding = np.zeros(1024, dtype=np.float32)
            self.embedding_cache[sequence] = embedding
            return embedding

class GNNDatasetWithProtT5(GNNDataset):
    """增强的GNN数据集 - 集成ProtT5嵌入"""
    
    def __init__(self, root, types, use_surface=True, use_masif=True, use_prot_t5=True, 
                 prot_t5_model_path="/home/lww/prot_t5_model", device='cpu'):
        
        self.use_prot_t5 = use_prot_t5
        self.prot_t5_generator = None
        self.protein_embeddings = {}
        
        if use_prot_t5:
            self.prot_t5_generator = ProtT5EmbeddingGenerator(prot_t5_model_path, device)
            # 尝试加载模型，如果失败则使用预计算嵌入
            model_loaded = self.prot_t5_generator.load_model()
            
            if not model_loaded:
                # 尝试加载预计算的嵌入
                self._load_precomputed_embeddings(root)
        
        # 调用父类初始化
        super().__init__(root, types, use_surface, use_masif)
        
        print(f"✅ 数据集初始化完成")
        print(f"   - 表面特征: {use_surface}")
        print(f"   - MaSIF特征: {use_masif}")
        print(f"   - ProtT5嵌入: {use_prot_t5}")
    
    def _load_precomputed_embeddings(self, root):
        """加载预计算的ProtT5嵌入"""
        embedding_paths = [
            os.path.join(root, "protein_embeddings.npy"),
            os.path.join(root, "saved_protein_data", "protein_embeddings.npy"),
            "saved_protein_data/protein_embeddings.npy",
            "protein_embeddings.npy"
        ]
        
        protein_list_paths = [
            os.path.join(root, "protein_list.pkl"),
            os.path.join(root, "saved_protein_data", "protein_list.pkl"),
            "saved_protein_data/protein_list.pkl",
            "protein_list.pkl"
        ]
        
        for emb_path, prot_path in zip(embedding_paths, protein_list_paths):
            try:
                if os.path.exists(emb_path) and os.path.exists(prot_path):
                    embeddings = np.load(emb_path)
                    with open(prot_path, 'rb') as f:
                        protein_list = pickle.load(f)
                    
                    # 创建蛋白质序列到嵌入的映射
                    for protein, embedding in zip(protein_list, embeddings):
                        self.protein_embeddings[protein] = embedding.astype(np.float32)
                    
                    print(f"✅ 加载预计算ProtT5嵌入: {len(self.protein_embeddings)} 个蛋白质")
                    return True
                    
            except Exception as e:
                print(f"⚠️ 加载预计算嵌入失败 ({emb_path}): {e}")
                continue
        
        print("⚠️ 未找到预计算的ProtT5嵌入，将使用随机初始化")
        return False
    
    def get_protein_embedding(self, protein_sequence):
        """获取蛋白质的ProtT5嵌入"""
        if not self.use_prot_t5:
            return None
        
        # 首先检查预计算的嵌入
        if protein_sequence in self.protein_embeddings:
            return self.protein_embeddings[protein_sequence]
        
        # 如果有ProtT5生成器，使用它生成嵌入
        if self.prot_t5_generator:
            embedding = self.prot_t5_generator.get_embedding(protein_sequence)
            self.protein_embeddings[protein_sequence] = embedding
            return embedding
        
        # 最后的fallback：随机嵌入
        embedding = np.random.normal(0, 0.1, 1024).astype(np.float32)
        self.protein_embeddings[protein_sequence] = embedding
        return embedding
    
    def get(self, idx):
        """重写get方法以包含ProtT5嵌入"""
        # 获取原始数据
        data = super().get(idx)
        
        # 添加ProtT5嵌入
        if self.use_prot_t5:
            # 从数据中获取蛋白质序列（需要根据实际数据结构调整）
            # 这里假设蛋白质序列存储在某个地方，需要根据实际情况修改
            try:
                # 尝试从文件名或其他方式获取蛋白质序列
                # 这是一个占位符实现，需要根据实际数据结构调整
                protein_sequence = self._get_protein_sequence_for_idx(idx)
                
                if protein_sequence:
                    prot_t5_embedding = self.get_protein_embedding(protein_sequence)
                    data.prot_t5_embedding = torch.tensor(prot_t5_embedding, dtype=torch.float32)
                else:
                    # 如果无法获取序列，使用零向量
                    data.prot_t5_embedding = torch.zeros(1024, dtype=torch.float32)
                    
            except Exception as e:
                print(f"⚠️ 获取ProtT5嵌入失败 (idx={idx}): {e}")
                data.prot_t5_embedding = torch.zeros(1024, dtype=torch.float32)
        
        return data
    
    def _get_protein_sequence_for_idx(self, idx):
        """根据索引获取蛋白质序列 - 需要根据实际数据结构实现"""
        # 这是一个占位符方法，需要根据实际的数据存储方式来实现
        # 可能需要读取CSV文件或其他数据源来获取蛋白质序列
        
        try:
            # 尝试从数据文件中读取蛋白质序列
            data_file = os.path.join(self.root, 'raw', 'data.csv')
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                # 根据实际的数据结构调整这里的逻辑
                # 这里假设有一个protein_sequence列
                if idx < len(df) and 'protein_sequence' in df.columns:
                    return df.iloc[idx]['protein_sequence']
            
            return None
            
        except Exception as e:
            print(f"⚠️ 读取蛋白质序列失败: {e}")
            return None

def create_enhanced_dataloaders(dataset='kiba', batch_size=256, use_prot_t5=True, 
                               prot_t5_model_path="/home/lww/prot_t5_model", device='cpu'):
    """创建增强的数据加载器"""
    
    fpath = os.path.join('/home/lww/learn_project/MGraphDTA-dev/regression/data', dataset)
    
    print(f"🔄 创建增强数据加载器...")
    print(f"   - 数据集: {dataset}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - ProtT5嵌入: {use_prot_t5}")
    
    # 创建训练和测试数据集
    train_set = GNNDatasetWithProtT5(
        fpath, types='train', 
        use_surface=True, use_masif=True, use_prot_t5=use_prot_t5,
        prot_t5_model_path=prot_t5_model_path, device=device
    )
    
    test_set = GNNDatasetWithProtT5(
        fpath, types='test1',
        use_surface=True, use_masif=True, use_prot_t5=use_prot_t5,
        prot_t5_model_path=prot_t5_model_path, device=device
    )
    
    # 创建数据加载器
    from torch_geometric.data import DataLoader
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    print(f"✅ 数据加载器创建完成")
    print(f"   - 训练集: {len(train_set)} 样本")
    print(f"   - 测试集: {len(test_set)} 样本")
    
    return train_loader, test_loader

if __name__ == '__main__':
    # 测试增强数据加载器
    train_loader, test_loader = create_enhanced_dataloaders(
        dataset='kiba', 
        batch_size=32,  # 小批次用于测试
        use_prot_t5=True,
        device='cpu'
    )
    
    print("🧪 测试数据加载...")
    for i, data in enumerate(train_loader):
        print(f"批次 {i+1}:")
        print(f"  - 分子特征: {data.x.shape}")
        print(f"  - 边索引: {data.edge_index.shape}")
        if hasattr(data, 'prot_t5_embedding'):
            print(f"  - ProtT5嵌入: {data.prot_t5_embedding.shape}")
        print(f"  - 标签: {data.y.shape}")
        
        if i >= 2:  # 只测试前3个批次
            break
    
    print("✅ 数据加载测试完成")
