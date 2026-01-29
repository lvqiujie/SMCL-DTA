#!/usr/bin/env python3
"""
生成KIBA数据集的ProtT5嵌入
基于split_cold.ipynb中的预处理代码
"""

import os
import pandas as pd
import numpy as np
import torch
import pickle
from transformers import T5EncoderModel, T5Tokenizer
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def get_T5_model():
    """加载ProtT5模型和tokenizer"""
    model_path = "/home/lww/prot_t5_model"
    
    try:
        print(f"🔄 加载ProtT5模型从: {model_path}")
        model = T5EncoderModel.from_pretrained(model_path, local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True)
        
        model = model.to(device)
        model.eval()
        
        print("✅ ProtT5模型加载成功")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ ProtT5模型加载失败: {e}")
        return None, None

def get_embedding(sequence, model, tokenizer):
    """获取单个蛋白质序列的嵌入向量"""
    # 处理特殊氨基酸
    sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X')

    # 检测是使用ProtT5还是ProtBERT
    is_t5 = isinstance(model, T5EncoderModel)

    if is_t5:
        # 氨基酸之间加空格 (T5需要)
        sequence = ' '.join(list(sequence))

    # 编码序列
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 获取嵌入向量
    with torch.no_grad():
        if is_t5:
            embedding_repr = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            # T5返回的是last_hidden_state
            emb = embedding_repr.last_hidden_state.mean(dim=1)  # 对所有token进行平均
        else:
            # ProtBERT
            embedding_repr = model(**inputs)
            # 使用[CLS]标记的表示
            emb = embedding_repr.last_hidden_state[:, 0, :]

    return emb.cpu().numpy().squeeze()

def process_protein_set(protein_list, model, tokenizer, batch_size=32):
    """批量处理蛋白质序列并生成嵌入向量"""
    embeddings = []
    failed_count = 0

    # 批量处理以提高效率
    for i in range(0, len(protein_list), batch_size):
        batch = protein_list[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(protein_list) + batch_size - 1) // batch_size}")

        for protein in batch:
            try:
                embedding = get_embedding(protein, model, tokenizer)
                embeddings.append(embedding)
            except Exception as e:
                print(f"⚠️ Error processing protein (length={len(protein)}): {e}")
                failed_count += 1
                # 添加一个零向量作为占位符
                dim = 1024 if isinstance(model, T5EncoderModel) else 768
                embeddings.append(np.zeros(dim, dtype=np.float32))

    print(f"✅ 处理完成: {len(embeddings)} 个嵌入, {failed_count} 个失败")
    return np.array(embeddings)

def run_analysis(df):
    """处理蛋白质列表并进行聚类分析"""
    # 提取蛋白质序列集合
    drug_set = set()
    protein_set = set()
    for i in range(len(df)):
        drug_set.add(df.loc[i, 'compound_iso_smiles'])
        protein_set.add(df.loc[i, 'target_sequence'])

    print(f"药物数量: {len(drug_set)}")
    print(f"蛋白质数量: {len(protein_set)}")

    # 将集合转换为列表
    protein_list = list(protein_set)
    
    return protein_list

def create_protein_to_embedding_mapping(df, embeddings, protein_list):
    """创建蛋白质序列到嵌入的映射"""
    # 创建序列到嵌入的映射
    protein_to_embedding = {}
    for protein, embedding in zip(protein_list, embeddings):
        protein_to_embedding[protein] = embedding.astype(np.float32)
    
    # 创建索引到嵌入的映射 (用于训练时快速查找)
    index_to_embedding = {}
    for i, row in df.iterrows():
        protein_seq = row['target_sequence']
        if protein_seq in protein_to_embedding:
            index_to_embedding[i] = protein_to_embedding[protein_seq]
        else:
            # 如果找不到，使用零向量
            index_to_embedding[i] = np.zeros(1024, dtype=np.float32)
    
    return protein_to_embedding, index_to_embedding

def main():
    print("🚀 生成KIBA数据集的ProtT5嵌入")
    print("=" * 60)
    
    # 1. 加载KIBA数据
    data_path = '/home/lww/learn_project/MGraphDTA-dev/regression/data/kiba/raw/data.csv'
    print(f"📊 加载数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    print(f"✅ 数据加载完成: {len(df)} 条记录")
    
    # 2. 提取蛋白质序列
    print("🔍 提取蛋白质序列...")
    protein_list = run_analysis(df)
    
    # 3. 加载ProtT5模型
    model, tokenizer = get_T5_model()
    if model is None:
        print("❌ 无法加载ProtT5模型")
        return False
    
    # 4. 生成嵌入
    print("🧬 生成ProtT5嵌入...")
    embeddings = process_protein_set(protein_list, model, tokenizer, batch_size=16)
    
    # 5. 创建映射
    print("🗺️ 创建蛋白质序列到嵌入的映射...")
    protein_to_embedding, index_to_embedding = create_protein_to_embedding_mapping(
        df, embeddings, protein_list
    )
    
    # 6. 保存结果
    output_dir = "kiba_prot_t5_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存嵌入数组
    embeddings_path = os.path.join(output_dir, "protein_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"✅ 嵌入数组已保存: {embeddings_path}")
    
    # 保存蛋白质列表
    protein_list_path = os.path.join(output_dir, "protein_list.pkl")
    with open(protein_list_path, 'wb') as f:
        pickle.dump(protein_list, f)
    print(f"✅ 蛋白质列表已保存: {protein_list_path}")
    
    # 保存序列到嵌入的映射
    seq_to_emb_path = os.path.join(output_dir, "protein_to_embedding.pkl")
    with open(seq_to_emb_path, 'wb') as f:
        pickle.dump(protein_to_embedding, f)
    print(f"✅ 序列映射已保存: {seq_to_emb_path}")
    
    # 保存索引到嵌入的映射
    idx_to_emb_path = os.path.join(output_dir, "index_to_embedding.pkl")
    with open(idx_to_emb_path, 'wb') as f:
        pickle.dump(index_to_embedding, f)
    print(f"✅ 索引映射已保存: {idx_to_emb_path}")
    
    # 7. 验证结果
    print("\n📊 生成结果验证:")
    print(f"   - 蛋白质数量: {len(protein_list)}")
    print(f"   - 嵌入维度: {embeddings.shape}")
    print(f"   - 数据记录数: {len(df)}")
    print(f"   - 索引映射数: {len(index_to_embedding)}")
    
    # 检查嵌入质量
    non_zero_embeddings = np.sum(np.any(embeddings != 0, axis=1))
    print(f"   - 非零嵌入: {non_zero_embeddings}/{len(embeddings)} ({non_zero_embeddings/len(embeddings)*100:.1f}%)")
    
    print("\n🎉 ProtT5嵌入生成完成!")
    print(f"📁 输出目录: {output_dir}")
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print("\n✅ 可以继续修复train_dual_no.py中的ProtT5集成")
    else:
        print("\n❌ 嵌入生成失败，请检查错误信息")
