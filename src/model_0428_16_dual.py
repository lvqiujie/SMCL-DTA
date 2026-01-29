import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
import random
from torch_geometric.data import Data
import math
from torch_geometric.utils import subgraph, to_dense_adj
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from torch.nn.utils.weight_norm import weight_norm
'''
MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-target binding affinity Prediction
'''


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)

class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x


class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))
        return data


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features
        data = self.conv1(data)
        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = gnn.global_mean_pool(data.x, data.batch)
        x = self.classifer(x)
        return x


class IntelligentMasking(nn.Module):
    """GraphMAE风格的智能掩码策略，增加多种图增强方法"""

    def __init__(self, mask_rate=0.15, mask_strategy='degree', replace_rate=0.1):
        super().__init__()
        self.mask_rate = mask_rate
        self.mask_strategy = mask_strategy
        self.replace_rate = replace_rate  # 替换掩码的比例（用于随机特征替换）

    def compute_node_importance(self, data):
        """计算节点重要性分数"""
        if self.mask_strategy == 'degree':
            # 基于节点度数计算重要性
            edge_index = data.edge_index
            num_nodes = data.x.size(0)
            degrees = degree(edge_index[0], num_nodes=num_nodes) + degree(edge_index[1], num_nodes=num_nodes)
            return degrees
        elif self.mask_strategy == 'betweenness':
            # 简化的中心性近似计算
            edge_index = data.edge_index
            num_nodes = data.x.size(0)
            adj = to_dense_adj(edge_index)[0]
            # 使用2步邻居作为中心度的近似
            hop_2_adj = torch.mm(adj, adj)
            centrality = hop_2_adj.sum(dim=1)
            return centrality
        elif self.mask_strategy == 'pagerank':
            # 简化的PageRank近似
            edge_index = data.edge_index
            num_nodes = data.x.size(0)
            # adj = to_dense_adj(edge_index)[0]
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
            # 归一化邻接矩阵
            degrees = adj.sum(dim=1).clamp(min=1)
            norm_adj = adj / degrees.view(-1, 1)
            # 两次迭代近似PageRank
            pr = torch.ones(num_nodes, device=adj.device) / num_nodes
            for _ in range(2):
                pr = 0.85 * torch.mv(norm_adj, pr) + 0.15 / num_nodes
            return pr
        else:
            # 随机策略：返回均匀分布
            return None

    def apply_masking(self, data):
        """应用掩码策略到图数据"""
        device = data.x.device
        masked_data = data.clone()

        # 获取节点数量
        num_nodes = data.x.size(0)
        mask_num = max(1, int(num_nodes * self.mask_rate))

        # 计算节点重要性
        importance = self.compute_node_importance(data)

        if importance is not None:
            # 基于重要性进行采样
            prob = F.softmax(importance, dim=0)
            masked_indices = torch.multinomial(prob, mask_num, replacement=False)
        else:
            # 随机采样
            masked_indices = torch.randperm(num_nodes, device=device)[:mask_num]

        # 创建掩码
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[masked_indices] = True

        # 应用掩码和特征替换
        replace_num = int(mask_num * self.replace_rate)
        if replace_num > 0:
            replace_indices = masked_indices[:replace_num]
            zero_indices = masked_indices[replace_num:]

            # 随机特征替换
            rand_features = torch.randn_like(masked_data.x[replace_indices])
            masked_data.x[replace_indices] = rand_features

            # 零掩码
            masked_data.x[zero_indices] = 0
        else:
            # 全部使用零掩码
            masked_data.x[masked_indices] = 0

        return masked_data, mask
    
    def drop_nodes(self, data, drop_rate=0.1):
        """随机删除部分节点的特征
        
        Args:
            data: 输入数据
            drop_rate: 要删除的节点比例
            
        Returns:
            增强后的数据，掩码
        """
        device = data.x.device
        masked_data = data.clone()
        node_num = masked_data.x.size(0)
        
        # 计算要删除的节点数量
        drop_num = max(1, int(node_num * drop_rate))
        
        # 随机选择要删除的节点
        idx_drop = torch.randperm(node_num, device=device)[:drop_num]
        
        # 创建掩码
        mask = torch.zeros(node_num, dtype=torch.bool, device=device)
        mask[idx_drop] = True
        
        # 将选择的节点特征置零
        masked_data.x[idx_drop] = 0
        
        return masked_data, mask
    
    def permute_edges(self, data, permute_rate=0.1):
        """随机置换部分边的连接
        
        Args:
            data: 输入数据
            permute_rate: 要置换的边比例
            
        Returns:
            增强后的数据，掩码
        """
        device = data.edge_index.device
        masked_data = data.clone()
        edge_num = masked_data.edge_index.size(1)
        node_num = masked_data.x.size(0)
        
        # 计算要置换的边数量
        permute_num = max(1, int(edge_num * permute_rate))
        
        # 随机选择要置换的边
        idx_permute = torch.randperm(edge_num, device=device)[:permute_num]
        
        # 创建节点掩码 - 默认为空
        mask = torch.zeros(node_num, dtype=torch.bool, device=device)
        
        # 对选定的边，随机置换源节点或目标节点
        affected_nodes = set()
        for idx in idx_permute:
            if random.random() < 0.5:  # 50%概率置换源节点
                old_node = masked_data.edge_index[0, idx].item()
                new_node = torch.randint(0, node_num, (1,), device=device).item()
                masked_data.edge_index[0, idx] = new_node
                affected_nodes.add(old_node)
                affected_nodes.add(new_node)
            else:  # 50%概率置换目标节点
                old_node = masked_data.edge_index[1, idx].item()
                new_node = torch.randint(0, node_num, (1,), device=device).item()
                masked_data.edge_index[1, idx] = new_node
                affected_nodes.add(old_node)
                affected_nodes.add(new_node)
        
        # 将受影响的节点添加到掩码中
        for node in affected_nodes:
            if node < node_num:  # 安全检查
                mask[node] = True
        
        return masked_data, mask
    
    def extract_subgraph(self, data, keep_rate=0.8):
        """随机提取子图
        
        Args:
            data: 输入数据
            keep_rate: 要保留的节点比例
            
        Returns:
            增强后的数据，掩码
        """
        device = data.x.device
        masked_data = data.clone()
        node_num = masked_data.x.size(0)
        
        # 计算要保留的节点数量
        keep_num = max(3, int(node_num * keep_rate))
        
        # 随机选择要保留的节点
        idx_keep = torch.randperm(node_num, device=device)[:keep_num]
        
        # 创建掩码，用于过滤节点和边
        keep_mask = torch.zeros(node_num, dtype=torch.bool, device=device)
        keep_mask[idx_keep] = True
        
        # 保留所选节点相关的边
        edge_mask = keep_mask[masked_data.edge_index[0]] & keep_mask[masked_data.edge_index[1]]
        
        # 更新边索引
        masked_data.edge_index = masked_data.edge_index[:, edge_mask]
        
        # 更新边属性（如果存在）
        if hasattr(masked_data, 'edge_attr') and masked_data.edge_attr is not None:
            masked_data.edge_attr = masked_data.edge_attr[edge_mask]
        
        # 对于未选择的节点，将其特征置零（保留节点数量不变，只是将特征置零）
        mask = ~keep_mask  # 掩码为未保留的节点
        masked_data.x[mask] = 0
        
        return masked_data, mask

    def __call__(self, data, aug_type=None):
        """对图数据应用增强
        
        Args:
            data: 输入图数据
            aug_type: 增强类型，如果为None则随机选择
                      0: 智能掩码
                      1: 删除节点
                      2: 扰动边
                      3: 子图采样
        
        Returns:
            增强后的数据，掩码
        """
        if aug_type is None:
            aug_type = random.randint(0, 3)
            
        if aug_type == 0:
            # 智能掩码
            return self.apply_masking(data)
        elif aug_type == 1:
            # 删除节点
            return self.drop_nodes(data, drop_rate=self.mask_rate)
        elif aug_type == 2:
            # 扰动边
            return self.permute_edges(data, permute_rate=self.mask_rate)
        else:
            # 子图采样
            return self.extract_subgraph(data, keep_rate=1.0 - self.mask_rate)

# 交叉注意力融合模块
class CrossAttention(nn.Module):
    def __init__(self, protein_dim, ligand_dim, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 投影层
        self.protein_proj = nn.Linear(protein_dim, embed_dim)
        self.ligand_proj = nn.Linear(ligand_dim, embed_dim)
        self.protein_ln = nn.LayerNorm(protein_dim)
        self.ligand_ln = nn.LayerNorm(ligand_dim)

        # 注意力投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, protein_x, ligand_x):
        batch_size = protein_x.size(0)

        # 应用层标准化
        protein_x = self.protein_ln(protein_x)
        ligand_x = self.ligand_ln(ligand_x)

        # 线性投影
        protein_proj = self.protein_proj(protein_x)  # [batch_size, protein_len, embed_dim]
        ligand_proj = self.ligand_proj(ligand_x)  # [batch_size, ligand_len, embed_dim]

        # 多头注意力计算
        # 使用ligand作为query，protein作为key和value
        q = self.q_proj(ligand_proj).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(protein_proj).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(protein_proj).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = attn_weights @ v  # [batch_size, num_heads, ligand_len, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # 双向交叉注意力（可选）
        # 使用protein作为query，ligand作为key和value
        q_p = self.q_proj(protein_proj).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_l = self.k_proj(ligand_proj).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_l = self.v_proj(ligand_proj).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights_p = (q_p @ k_l.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights_p = F.softmax(attn_weights_p, dim=-1)
        attn_weights_p = self.dropout(attn_weights_p)

        attn_output_p = attn_weights_p @ v_l  # [batch_size, num_heads, protein_len, head_dim]
        attn_output_p = attn_output_p.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        attn_output_p = self.out_proj(attn_output_p)

        # 计算全局表示
        ligand_attn = torch.mean(attn_output, dim=1)  # [batch_size, embed_dim]
        protein_attn = torch.mean(attn_output_p, dim=1)  # [batch_size, embed_dim]

        # 融合表示
        fused_embedding = torch.cat([ligand_attn, protein_attn], dim=1)  # [batch_size, 2*embed_dim]

        return fused_embedding

class DenseBilinear(nn.Module):
    """稠密双线性层"""
    def __init__(self, protein_dim, ligand_dim, output_dim, dropout=0.1):
        super(DenseBilinear, self).__init__()

        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.output_dim = output_dim

        # 双线性投影矩阵
        self.bilinear = nn.Bilinear(protein_dim, ligand_dim, output_dim)

        # 线性投影
        self.protein_proj = nn.Linear(protein_dim, output_dim)
        self.ligand_proj = nn.Linear(ligand_dim, output_dim)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    def forward(self, protein_x, ligand_x):
        """
        Args:
            protein_x: 蛋白质特征 [batch_size, protein_dim]
            ligand_x: 配体特征 [batch_size, ligand_dim]

        Returns:
            融合特征 [batch_size, output_dim]
        """
        # 双线性交互
        bilinear_out = self.bilinear(protein_x, ligand_x)

        # 线性投影
        protein_proj = self.protein_proj(protein_x)
        ligand_proj = self.ligand_proj(ligand_x)

        # 合并所有特征（残差连接）
        output = bilinear_out + protein_proj + ligand_proj
        output = self.output_layer(output)

        return output

def nt_xent_loss(z1, z2, temperature=0.1, normalize=True):
    """
    MolCLR风格的NT-Xent对比损失

    参数:
        z1, z2: 两个视图的表示 [batch_size, dim]
        temperature: 温度参数
        normalize: 是否对输入进行L2归一化

    返回:
        对比损失值
    """
    if normalize:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    # 将两个视图的表示拼接在一起
    features = torch.cat([z1, z2], dim=0)

    # 计算点积相似度矩阵
    similarity_matrix = torch.matmul(features, features.T)

    # 除以温度参数
    similarity_matrix = similarity_matrix / temperature

    # 构建标签: [0,...,batch_size-1,batch_size,...,2*batch_size-1]
    labels = torch.cat([
        torch.arange(batch_size, device=z1.device) + batch_size,  # 第一个视图对应第二个视图
        torch.arange(batch_size, device=z1.device)  # 第二个视图对应第一个视图
    ], dim=0)

    # 去除自相似度
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    # 将对角线元素设为大的负值，使exp接近0
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    # 应用softmax得到归一化相似度
    similarity_matrix = F.log_softmax(similarity_matrix, dim=1)

    # 提取正样本对的相似度
    similarity_matrix = similarity_matrix.masked_select(
        F.one_hot(labels, num_classes=2 * batch_size).bool()
    ).view(2 * batch_size, 1)

    # 计算负对数似然
    loss = -similarity_matrix.mean()

    return loss

class MLPSurfaceEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=32):
        super(MLPSurfaceEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 简单的共享 MLP (两层)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 最终的 MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, surface_data, num_points=None):
        # --- 1. Reshape 输入 ---
        if surface_data.dim() == 2:
            # 输入: [batch_size * N, C]
            if num_points is None:
                raise ValueError("num_points must be provided when input is 2D ([B*N, C])")
            if surface_data.size(0) == 0: # 处理空输入
                 return torch.zeros((0, self.output_dim), device=surface_data.device)

            # 检查总点数是否能被 num_points 整除
            total_elements = surface_data.size(0)
            if total_elements % num_points != 0:
                 raise ValueError(f"Input size {total_elements} is not divisible by num_points {num_points}")

            batch_size = total_elements // num_points
            # 使用传入的 num_points 进行 reshape
            surface = surface_data.view(batch_size, num_points, self.input_dim)

        elif surface_data.dim() == 3:
            # 输入: [batch_size, N, C]
            batch_size, current_num_points, channels = surface_data.size()
            if channels != self.input_dim:
                 raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {channels}")
            # N (current_num_points) 可以从输入直接获得
            surface = surface_data
            if batch_size == 0: # 处理空输入
                return torch.zeros((0, self.output_dim), device=surface_data.device)
            # 可选：如果提供了 num_points，可以进行检查
            if num_points is not None and current_num_points != num_points:
                 print(f"Warning: Provided num_points ({num_points}) does not match input tensor's points ({current_num_points}). Using value from tensor.")
                 # 或者 raise ValueError("Mismatch between provided num_points and input tensor shape")
                 num_points = current_num_points # 以张量形状为准

        else:
            raise ValueError(f"Unsupported input dimension: {surface_data.dim()}. Expected 2 or 3.")

        # --- 后续步骤与之前相同 ---

        # --- 2. 应用共享 MLP ---
        x = surface.permute(0, 2, 1)  # 变形为 [B, C, N]
        point_features = self.shared_mlp(x) # 输出 [B, hidden_dim, N]

        # --- 3. 聚合特征 (Max Pooling) ---
        aggregated_features = torch.max(point_features, dim=2)[0] # 输出 [B, hidden_dim]

        # --- 4. 应用最终 MLP ---
        global_features = self.final_mlp(aggregated_features) # 输出 [B, output_dim]

        return global_features

class MGraphDTA(nn.Module):
    def __init__(self, block_num, vocab_protein_size, embedding_size=128, filter_num=32, out_dim=1,
                 mask_rate=0.12, proj_dim=128, temperature=0.1, disable_masking=False, cl_mode='instance',
                 cl_similarity_threshold=0.2, use_surface=False):
        super().__init__()
        self.protein_encoder = TargetRepresentation(block_num, vocab_protein_size, embedding_size)
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num * 3, block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])

        drug_dim = 96+32+32
        target_dim = 96 + 32
        h_dim = 128
        n_heads = 6
        # 计算输出特征维度 - 修正实际输出维度
        protein_feat_dim = 96  # TargetRepresentation的输出维度固定为96
        ligand_feat_dim = filter_num * 3  # 96 from GraphDenseNet (32*3)

        # self.drug_target_fusion = DenseBilinear(target_dim, drug_dim, 128, 0.2)
        # self.drug_target_fusion = DenseBilinear(target_dim, drug_dim, 192, 0.2)
        # self.drug_target_fusion = weight_norm(
        #     BANLayer(v_dim=drug_dim, q_dim=target_dim, h_dim=h_dim, h_out=n_heads, k=3),
        #     name='h_mat', dim=None)
        
        print(f"初始特征维度: protein_feat_dim={protein_feat_dim}, ligand_feat_dim={ligand_feat_dim}")

        # 表面特征处理
        self.use_surface = use_surface
        if use_surface:
            # 配体全局特征编码器
            self.ligand_global_encoder = nn.Sequential(
                nn.Linear(264, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 16)  # 减小输出维度
            )

            # 调整feature_dim大小
            feature_dim = protein_feat_dim + ligand_feat_dim + 16 + 16 + 16  # 原始特征 + 三个表面特征: 96 + 96 + 32*3 = 288
        else:
            feature_dim = protein_feat_dim + ligand_feat_dim  # 原始特征: 96 + 96 = 192

        # 保存特征维度以便调试
        self.feature_dim = feature_dim
        print(f"模型特征维度: {feature_dim}")

        # 分类器
        # self.classifier = nn.Sequential(
        #     nn.Linear(feature_dim, 1024),
        #     nn.ReLU(),
        #     # nn.BatchNorm1d(1024),
        #     nn.Dropout(0.25),
        #     # nn.Linear(1024, 1024),
        #     # nn.ReLU(),
        #     # nn.BatchNorm1d(1024),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     # nn.BatchNorm1d(512),
        #     nn.Dropout(0.25),
        #     nn.Linear(512, out_dim)
        # )
        # feature_dim += 120
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

        self.ligand_surface_encoder = MLPSurfaceEncoder(input_dim=6, output_dim=16)
        self.protein_surface_encoder = MLPSurfaceEncoder(input_dim=9, output_dim=16)
        # 投影头简化为直接降维
        # self.projection_head = nn.Sequential(
        #     nn.Linear(feature_dim, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, proj_dim)
        # )

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
        )


        # 配置参数
        self.mask_rate = mask_rate
        self.temperature = temperature
        self.use_contrastive = False
        self.disable_masking = disable_masking
        self.cl_mode = cl_mode
        self.cl_similarity_threshold = cl_similarity_threshold

        # 添加智能掩码组件
        self.intelligent_masking = IntelligentMasking(
            mask_rate=mask_rate,
            mask_strategy='pagerank',
            replace_rate=0.1
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def get_embeddings(self, data, apply_masking=False):
        """获取分子和蛋白质的融合特征表示"""
        data_processed = data.clone()
        
        # 如果需要掩码，并且处于训练模式
        if apply_masking and self.training and not self.disable_masking:
            # 检查是否指定了增强类型
            # aug_type = None
            # if hasattr(data, 'aug_type'):
            #     aug_type = data.aug_type
            aug_type = 0
            # 应用智能掩码策略，使用指定的增强类型
            masked_data, _ = self.intelligent_masking(data_processed, aug_type=aug_type)
            data_processed = masked_data

        # 提取基本特征
        target = data_processed.target
        protein_x = self.protein_encoder(target)
        ligand_x = self.ligand_encoder(data_processed)
        
        if self.use_surface:
            try:
                # 安全地获取表面特征
                ligand_surface_exists = hasattr(data, 'ligand_surface') and data.ligand_surface is not None
                protein_surface_exists = hasattr(data, 'protein_surface') and data.protein_surface is not None
                
                if ligand_surface_exists and protein_surface_exists:
                    # 编码表面特征
                    ligand_surface_x = self.ligand_surface_encoder(data.ligand_surface, num_points=80)
                    protein_surface_x = self.protein_surface_encoder(data.protein_surface, num_points=512)
                    
                    # 编码全局特征
                    if hasattr(data, 'ligand_global') and data.ligand_global is not None:
                        ligand_global_x = self.ligand_global_encoder(data.ligand_global.view(ligand_x.size(0), -1))
                    else:
                        ligand_global_x = torch.zeros(ligand_x.size(0), 8, device=ligand_x.device)
                    
                    # 融合所有特征
                    x = torch.cat([protein_x, protein_surface_x], dim=-1)  # 96+32 ->
                    y = torch.cat([ligand_x, ligand_surface_x, ligand_global_x], dim=-1)  # 96+32+32 ->
                    # x = self.drug_target_fusion(x, y)
                    x = torch.cat([x, y], dim=-1)  # 96+32+96+32+32=288
                    # x, att = self.drug_target_fusion(y, x)
                    # 调试输出
                    if not hasattr(self, '_debug_printed'):
                        print(f"特征形状：protein_x={protein_x.shape}, ligand_x={ligand_x.shape}, "
                              f"protein_surface_x={protein_surface_x.shape}, ligand_surface_x={ligand_surface_x.shape}, "
                              f"ligand_global_x={ligand_global_x.shape}, 合并后={x.shape}")
                        self._debug_printed = True
                else:
                    # 不使用表面特征
                    print("警告: 表面特征缺失，回退到基本特征")
                    x = torch.cat([protein_x, ligand_x], dim=-1)
            except Exception as e:
                # 异常处理，确保即使表面特征处理失败也能回退到基本特征
                print(f"处理表面特征时出错: {e}，回退到基本特征")
                x = torch.cat([protein_x, ligand_x], dim=-1)
        else:
            # 不使用表面特征
            x = torch.cat([protein_x, ligand_x], dim=-1)
            
        return x

    def forward(self, data, apply_masking=False):
        """前向传播"""
        # 获取特征表示
        x = self.get_embeddings(data, apply_masking=apply_masking)
        pred = self.classifier(x)

        return pred

    def fuse_embeddings(self, embeddings1, embeddings2, strategy='weighted'):
        """特征融合策略"""
        if strategy == 'weighted':
            # 使用可学习的权重进行加权融合
            alpha = torch.sigmoid(self.fusion_weight)
            fused = alpha * embeddings1 + (1 - alpha) * embeddings2
            # 添加LayerNorm以稳定训练
            layer_norm = nn.LayerNorm(fused.size(-1)).to(fused.device)
            fused = layer_norm(fused)
            return fused

        elif strategy == 'concat':
            # 拼接后降维
            concat_embeddings = torch.cat([embeddings1, embeddings2], dim=1)
            F.linear()
            return self.fusion_dim_reducer(concat_embeddings)

        else:  # 'average' - 最简单的策略
            return (embeddings1 + embeddings2) / 2

    def compute_contrastive_loss(self, embeddings1, embeddings2):
        """
        计算两个视图之间的对比损失，使用MolCLR风格的NT-Xent损失
        """
        # 基于cl_mode决定损失计算方式
        if self.cl_mode == 'instance':
            # 实例级对比学习 - 标准NT-Xent损失
            return nt_xent_loss(embeddings1, embeddings2, temperature=self.temperature)

        elif self.cl_mode == 'supervised':
            # 加入标签信息的对比学习
            # 在这种模式下，相似标签的样本应该有相似的表示
            if not hasattr(self, '_last_batch_labels'):
                # 如果没有保存标签信息，使用标准NT-Xent损失
                return nt_xent_loss(embeddings1, embeddings2, temperature=self.temperature)

            # 创建标签相似性矩阵
            labels = self._last_batch_labels
            label_sim = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()

            # 计算样本间相似度
            z1 = F.normalize(embeddings1, dim=1)
            z2 = F.normalize(embeddings2, dim=1)

            batch_size = z1.size(0)
            features = torch.cat([z1, z2], dim=0)
            sim_matrix = torch.matmul(features, features.T) / self.temperature

            # 去除自相似度
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
            sim_matrix = sim_matrix.masked_fill(mask, -9e15)

            # 计算损失
            exp_sim = torch.exp(sim_matrix)

            # 扩展标签相似性矩阵
            extended_label_sim = torch.zeros(2 * batch_size, 2 * batch_size, device=z1.device)
            extended_label_sim[:batch_size, batch_size:] = label_sim
            extended_label_sim[batch_size:, :batch_size] = label_sim.T

            # 使用标签相似性加权对比损失
            pos_weights = extended_label_sim
            pos_weights = pos_weights / pos_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

            loss = -torch.log(
                (exp_sim * pos_weights).sum(dim=1) / exp_sim.sum(dim=1)
            ).mean()

            return loss

        elif self.cl_mode == 'regression':
            # 回归问题的对比学习，基于预测值的相似性
            # 实现类似于CLDR (Contrastive Learning for Drug Response)的策略
            if not hasattr(self, '_last_batch_labels'):
                # 如果没有保存标签信息，使用标准NT-Xent损失
                return nt_xent_loss(embeddings1, embeddings2, temperature=self.temperature)

            # 获取当前批次的标签 (假设是连续值)
            labels = self._last_batch_labels.view(-1)

            # 计算标签差异的标准差
            label_diff = (labels.unsqueeze(1) - labels.unsqueeze(0)).abs()
            std = label_diff.std()

            # 基于标签差异构建相似性矩阵
            # 差异小于阈值的样本对被视为正样本对
            threshold = self.cl_similarity_threshold * std
            label_sim = (label_diff < threshold).float()

            # 计算样本间相似度
            z1 = F.normalize(embeddings1, dim=1)
            z2 = F.normalize(embeddings2, dim=1)

            batch_size = z1.size(0)
            features = torch.cat([z1, z2], dim=0)
            sim_matrix = torch.matmul(features, features.T) / self.temperature

            # 去除自相似度
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
            sim_matrix = sim_matrix.masked_fill(mask, -9e15)

            # 计算损失
            exp_sim = torch.exp(sim_matrix)

            # 扩展标签相似性矩阵
            extended_label_sim = torch.zeros(2 * batch_size, 2 * batch_size, device=z1.device)
            extended_label_sim[:batch_size, batch_size:] = label_sim
            extended_label_sim[batch_size:, :batch_size] = label_sim.T

            # 使用标签相似性加权对比损失
            pos_weights = extended_label_sim
            # 避免无正样本的情况
            row_sums = pos_weights.sum(dim=1, keepdim=True)
            pos_weights = torch.where(row_sums > 0, pos_weights / row_sums, torch.zeros_like(pos_weights))

            # 对于没有正样本的行，使用原始的NT-Xent策略
            no_pos_mask = (row_sums == 0).squeeze()
            if no_pos_mask.any():
                # 默认的正样本对位置
                default_pos = torch.cat([
                    torch.arange(batch_size, device=z1.device) + batch_size,
                    torch.arange(batch_size, device=z1.device)
                ])

                for i in torch.where(no_pos_mask)[0]:
                    pos_weights[i, default_pos[i]] = 1.0

            loss = -torch.log(
                (exp_sim * pos_weights).sum(dim=1) / exp_sim.sum(dim=1)
            ).mean()

            return loss

        # elif self.cl_mode == 'regression':
        #     stats = {}
        #     if self._last_batch_labels is None:
        #         print("回归对比模式下 _last_batch_labels 未设置。回退到实例对比损失。")
        #         loss = nt_xent_loss(embeddings1, embeddings2, temperature=self.temperature)
        #         # stats 保持为空
        #         return loss, stats  # 返回元组
        #
        #     labels = self._last_batch_labels.view(-1)
        #     batch_size_labels = labels.size(0)
        #     batch_size_embeds = embeddings1.size(0)
        #
        #     if batch_size_labels != batch_size_embeds:
        #         print(f"标签 ({batch_size_labels}) 和嵌入 ({batch_size_embeds}) 的批次大小不匹配。回退到实例对比损失。")
        #         loss = nt_xent_loss(embeddings1, embeddings2, temperature=self.temperature)
        #         return loss, stats  # 返回元组
        #     if batch_size_labels == 0:
        #         return torch.tensor(0.0, requires_grad=True, device=embeddings1.device), {}  # 返回元组
        #
        #     # --- 以下是回归对比损失的计算逻辑 ---
        #     std = torch.tensor(0.0, device=embeddings1.device)
        #     if batch_size_labels > 1:
        #         label_diff = (labels.unsqueeze(1) - labels.unsqueeze(0)).abs()
        #         std = label_diff.std()
        #
        #     actual_threshold_value = -1.0
        #     num_extra_pos_pairs_in_batch = 0.0
        #
        #     if torch.isnan(std) or std < 1e-6:
        #         label_sim = torch.zeros((batch_size_labels, batch_size_labels), device=embeddings1.device, dtype=torch.float)
        #     else:
        #         threshold_value = self.cl_similarity_threshold * std
        #         label_sim = (label_diff < threshold_value).float()
        #         actual_threshold_value = threshold_value.item()
        #         if batch_size_labels > 1:
        #             # 计算额外正样本对数量 (确保对角线被排除)
        #             label_sim_no_diag = label_sim.clone()
        #             label_sim_no_diag.fill_diagonal_(0)  # 移除对角线上的1
        #             num_extra_pos_pairs_in_batch = label_sim_no_diag.sum().item() / 2.0
        #
        #     z1_proj_norm = F.normalize(embeddings1, dim=1)
        #     z2_proj_norm = F.normalize(embeddings2, dim=1)
        #
        #     features_proj = torch.cat([z1_proj_norm, z2_proj_norm], dim=0)
        #
        #     sim_matrix_no_temp = torch.matmul(features_proj, features_proj.T)
        #     sim_matrix = sim_matrix_no_temp / self.temperature
        #
        #     mask_diag = torch.eye(2 * batch_size_embeds, dtype=torch.bool, device=embeddings1.device)
        #     sim_matrix = sim_matrix.masked_fill(mask_diag, -float('inf'))
        #     exp_sim = torch.exp(sim_matrix)
        #
        #     pos_weights = torch.zeros_like(sim_matrix, dtype=torch.float)
        #     # 1. 自身增强视图是正样本
        #     for i in range(batch_size_embeds):
        #         pos_weights[i, i + batch_size_embeds] = 1.0
        #         pos_weights[i + batch_size_embeds, i] = 1.0
        #
        #     # 2. 基于标签相似性的额外正样本
        #     if label_sim.shape[0] == batch_size_embeds and label_sim.shape[1] == batch_size_embeds:
        #         # 确保 label_sim 的对角线为0，避免将自身视为额外正样本
        #         label_sim_no_diag = label_sim.clone()
        #         label_sim_no_diag.fill_diagonal_(0)
        #         # 使用 torch.max 合并，确保自身增强的权重至少为1（如果label_sim对应位置也为1则保持）
        #         pos_weights[:batch_size_embeds, batch_size_embeds:] = torch.max(
        #             pos_weights[:batch_size_embeds, batch_size_embeds:], label_sim_no_diag  # 使用无对角线的label_sim
        #         )
        #         pos_weights[batch_size_embeds:, :batch_size_embeds] = torch.max(
        #             pos_weights[batch_size_embeds:, :batch_size_embeds], label_sim_no_diag.T  # 使用无对角线的label_sim
        #         )
        #
        #     row_sums = pos_weights.sum(dim=1, keepdim=True)
        #     safe_row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        #     pos_weights_normalized = pos_weights / safe_row_sums
        #
        #     numerator = (exp_sim * pos_weights_normalized).sum(dim=1)
        #     denominator = exp_sim.sum(dim=1)
        #
        #     epsilon = 1e-9
        #     loss = -torch.log(numerator / (denominator + epsilon) + epsilon)
        #
        #     if torch.isnan(loss).any() or torch.isinf(loss).any():
        #         print(f"回归对比损失计算中出现 NaN/Inf。Numerator: {numerator}, Denominator: {denominator}")
        #         loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        #     else:
        #         loss = loss.mean()
        #
        #     # 填充 stats 字典
        #     stats = {
        #         "std_label_diff": std.item() if isinstance(std, torch.Tensor) else std,
        #         "actual_abs_threshold": actual_threshold_value,
        #         "num_extra_pos_pairs_in_batch": num_extra_pos_pairs_in_batch
        #     }
        #     return loss, stats  # 返回元组
        else:
            # 默认使用标准NT-Xent损失
            return nt_xent_loss(embeddings1, embeddings2, temperature=self.temperature)


def calculate_uniformity_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    计算一个批次嵌入向量的均匀性损失 (L_reg)。

    参数:
        embeddings (torch.Tensor): 形状为 [batch_size, embedding_dim] 的嵌入向量。

    返回:
        torch.Tensor: 一个标量，代表计算出的均匀性损失。
    """
    # 步骤一：L2归一化
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)

    # 步骤二：找到最近邻
    dots = torch.mm(norm_embeddings, norm_embeddings.t())
    n = norm_embeddings.shape[0]
    dots.view(-1)[::(n + 1)].fill_(-1)
    _, nearest_neighbor_indices = torch.max(dots, 1)

    # 步骤三：计算到最近邻的距离
    pdist = nn.PairwiseDistance(p=2)
    nearest_neighbors = norm_embeddings[nearest_neighbor_indices]
    distances = pdist(norm_embeddings, nearest_neighbors)

    # 步骤四：计算最终损失
    batch_size = n
    # 乘以 batch_size 是可选的缩放，核心是 -log(distance)
    loss_uniform = -torch.log(batch_size * distances).mean()

    return loss_uniform

if __name__ == '__main__':
    import pickle

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=2).to(device)
    data = pickle.load(open('/home/lww/learn_project/DDI_DTA/data_batch.pkl', 'rb'))
    data = data.to(device)
    pred = model(data)
    print(model)