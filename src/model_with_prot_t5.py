#!/usr/bin/env python3
"""
ProtT5增强的MGraphDTA模型
在原有架构基础上集成ProtT5蛋白质嵌入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

# 导入原始模型
from model_0428_16_dual import MGraphDTA

class ProtT5FusionModule(nn.Module):
    """ProtT5特征融合模块"""
    
    def __init__(self, prot_t5_dim=1024, protein_dim=96, fusion_dim=128):
        super().__init__()
        
        self.prot_t5_dim = prot_t5_dim
        self.protein_dim = protein_dim
        self.fusion_dim = fusion_dim
        
        # ProtT5嵌入处理
        self.prot_t5_projector = nn.Sequential(
            nn.Linear(prot_t5_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # 原始蛋白质特征处理
        self.protein_projector = nn.Sequential(
            nn.Linear(protein_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )
        
        # 多模态融合
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合后的特征处理
        self.fusion_output = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, protein_dim)  # 输出维度与原始蛋白质特征一致
        )
        
    def forward(self, protein_features, prot_t5_embeddings):
        """
        Args:
            protein_features: [batch_size, protein_dim] 原始蛋白质特征
            prot_t5_embeddings: [batch_size, prot_t5_dim] ProtT5嵌入
        Returns:
            fused_features: [batch_size, protein_dim] 融合后的蛋白质特征
        """
        batch_size = protein_features.size(0)
        
        # 处理ProtT5嵌入
        prot_t5_proj = self.prot_t5_projector(prot_t5_embeddings)  # [batch_size, fusion_dim]
        
        # 处理原始蛋白质特征
        protein_proj = self.protein_projector(protein_features)    # [batch_size, fusion_dim]
        
        # 准备多头注意力输入 [batch_size, seq_len, embed_dim]
        # 这里seq_len=2，分别对应ProtT5和原始特征
        multi_modal_input = torch.stack([prot_t5_proj, protein_proj], dim=1)  # [batch_size, 2, fusion_dim]
        
        # 多头注意力融合
        fused_output, attention_weights = self.fusion_attention(
            multi_modal_input, multi_modal_input, multi_modal_input
        )  # [batch_size, 2, fusion_dim]
        
        # 聚合融合结果
        aggregated = fused_output.mean(dim=1)  # [batch_size, fusion_dim]
        
        # 输出处理
        final_features = self.fusion_output(aggregated)  # [batch_size, protein_dim]
        
        return final_features

class MGraphDTAWithProtT5(MGraphDTA):
    """集成ProtT5的增强MGraphDTA模型"""
    
    def __init__(self, num_features_mol, num_features_pro, embedding_size=128, 
                 filter_num=32, out_dim=1, mask_rate=0.05, temperature=0.1,
                 disable_masking=False, cl_mode='regression', cl_similarity_threshold=0.5,
                 use_surface=True, use_prot_t5=True, prot_t5_fusion_dim=128):
        
        # 初始化父类
        super().__init__(
            num_features_mol, num_features_pro, embedding_size, filter_num, out_dim,
            mask_rate, temperature, disable_masking, cl_mode, cl_similarity_threshold, use_surface
        )
        
        self.use_prot_t5 = use_prot_t5
        
        if use_prot_t5:
            # 添加ProtT5融合模块
            self.prot_t5_fusion = ProtT5FusionModule(
                prot_t5_dim=1024,
                protein_dim=num_features_pro,
                fusion_dim=prot_t5_fusion_dim
            )
            
            print(f"✅ ProtT5融合模块已添加")
            print(f"   - ProtT5维度: 1024")
            print(f"   - 蛋白质特征维度: {num_features_pro}")
            print(f"   - 融合维度: {prot_t5_fusion_dim}")
    
    def forward(self, data):
        """前向传播 - 集成ProtT5特征"""
        
        # 分子图编码 (保持原有逻辑)
        mol_x, mol_edge_index, mol_batch = data.x, data.edge_index, data.batch
        
        # 分子特征提取
        mol_x = self.mol_conv1(mol_x, mol_edge_index)
        mol_x = F.relu(mol_x)
        mol_x = self.mol_conv2(mol_x, mol_edge_index)
        mol_x = F.relu(mol_x)
        mol_x = self.mol_conv3(mol_x, mol_edge_index)
        
        # 分子图池化
        mol_x = torch.cat([global_mean_pool(mol_x, mol_batch), 
                          global_max_pool(mol_x, mol_batch)], dim=1)
        
        # 蛋白质特征处理
        if hasattr(data, 'target') and data.target is not None:
            pro_x = data.target
        else:
            # 如果没有target字段，使用默认处理
            batch_size = mol_x.size(0)
            pro_x = torch.randn(batch_size, self.num_features_pro, device=mol_x.device)
        
        # ProtT5特征融合
        if self.use_prot_t5 and hasattr(data, 'prot_t5_embedding'):
            prot_t5_emb = data.prot_t5_embedding
            
            # 确保批次维度匹配
            if prot_t5_emb.size(0) != pro_x.size(0):
                # 如果维度不匹配，进行调整
                if prot_t5_emb.dim() == 1:
                    prot_t5_emb = prot_t5_emb.unsqueeze(0).repeat(pro_x.size(0), 1)
                elif prot_t5_emb.size(0) == 1 and pro_x.size(0) > 1:
                    prot_t5_emb = prot_t5_emb.repeat(pro_x.size(0), 1)
            
            # 应用ProtT5融合
            pro_x = self.prot_t5_fusion(pro_x, prot_t5_emb)
        
        # 蛋白质特征编码
        pro_x = F.relu(self.pro_fc1(pro_x))
        pro_x = F.dropout(pro_x, training=self.training)
        pro_x = F.relu(self.pro_fc2(pro_x))
        pro_x = F.dropout(pro_x, training=self.training)
        pro_x = F.relu(self.pro_fc3(pro_x))
        
        # 分子-蛋白质特征融合
        combined = torch.cat([mol_x, pro_x], dim=1)
        
        # 最终预测
        combined = F.relu(self.fc1(combined))
        combined = F.dropout(combined, training=self.training)
        combined = F.relu(self.fc2(combined))
        combined = F.dropout(combined, training=self.training)
        output = self.out(combined)
        
        return output
    
    def get_feature_dimensions(self):
        """获取特征维度信息"""
        info = {
            'molecular_features': self.num_features_mol,
            'protein_features': self.num_features_pro,
            'embedding_size': self.embedding_size,
            'use_prot_t5': self.use_prot_t5
        }
        
        if self.use_prot_t5:
            info['prot_t5_dim'] = 1024
            info['fusion_dim'] = self.prot_t5_fusion.fusion_dim
        
        return info

def create_enhanced_model(num_features_mol=3, num_features_pro=97, use_prot_t5=True, **kwargs):
    """创建ProtT5增强的模型"""
    
    model = MGraphDTAWithProtT5(
        num_features_mol=num_features_mol,
        num_features_pro=num_features_pro,
        use_prot_t5=use_prot_t5,
        **kwargs
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️ ProtT5增强模型创建完成")
    print(f"   - 总参数: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    print(f"   - ProtT5集成: {use_prot_t5}")
    
    return model

if __name__ == '__main__':
    # 测试模型创建
    print("🧪 测试ProtT5增强模型...")
    
    # 创建模型
    model = create_enhanced_model(
        num_features_mol=3,
        num_features_pro=97,  # 96 + 1 (原始特征)
        embedding_size=128,
        filter_num=32,
        use_prot_t5=True
    )
    
    # 打印模型信息
    feature_info = model.get_feature_dimensions()
    print(f"\n📊 特征维度信息:")
    for key, value in feature_info.items():
        print(f"   - {key}: {value}")
    
    # 创建测试数据
    batch_size = 4
    num_nodes = 20
    
    # 模拟图数据
    x = torch.randn(num_nodes * batch_size, 3)
    edge_index = torch.randint(0, num_nodes * batch_size, (2, num_nodes * batch_size * 2))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    target = torch.randn(batch_size, 97)
    prot_t5_embedding = torch.randn(batch_size, 1024)
    y = torch.randn(batch_size, 1)
    
    # 创建数据对象
    from torch_geometric.data import Data
    test_data = Data(
        x=x,
        edge_index=edge_index,
        batch=batch,
        target=target,
        prot_t5_embedding=prot_t5_embedding,
        y=y
    )
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        output = model(test_data)
        print(f"\n🔄 前向传播测试:")
        print(f"   - 输入形状: 分子节点={x.shape}, 蛋白质={target.shape}, ProtT5={prot_t5_embedding.shape}")
        print(f"   - 输出形状: {output.shape}")
        print(f"   - 输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    print("✅ 模型测试完成")
