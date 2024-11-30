import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO

def load_sequences(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def generate_embeddings(sequences, tokenizer, model, device, batch_size=16):
    model.eval()
    embeddings = []
    sequence_ids = []
    sequence_list = []
    for seq_id, seq in sequences.items():
        sequence_ids.append(seq_id)
        sequence_list.append(seq)

    all_batch_ids = []  # 用于存储所有的序列 ID

    for i in tqdm(range(0, len(sequence_list), batch_size), desc="Generating Embeddings"):
        batch_sequences = sequence_list[i:i+batch_size]
        batch_ids = sequence_ids[i:i+batch_size]
        inputs = tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # 保持维度 [批次大小, 序列长度, 隐藏层维度]
            last_hidden_state = outputs.hidden_states[-1].permute(1, 0, 2)  # [batch_size, seq_length, hidden_dim]
            print(f"last_hidden_state shape: {last_hidden_state.shape}")  # 应该是 [batch_size, seq_length, hidden_dim]
            
            attention_mask = inputs['attention_mask']  # [batch_size, seq_length]
            attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_length, 1]
            print(f"attention_mask shape: {attention_mask.shape}")  # 应该是 [batch_size, seq_length, 1]
            
            # 计算嵌入，确保分母的维度与分子匹配
            numerator = (last_hidden_state * attention_mask).sum(dim=1)             # [batch_size, hidden_dim]
            denominator = attention_mask.sum(dim=1)                                 # [batch_size, 1]
            seq_embeddings = numerator / denominator                               # [batch_size, hidden_dim]
            print(f"seq_embeddings shape: {seq_embeddings.shape}")                  # 应该是 [batch_size, hidden_dim]
            
            embeddings.append(seq_embeddings.cpu())
            all_batch_ids.extend(batch_ids)  # 收集当前批次的序列 ID
    embeddings = torch.cat(embeddings, dim=0)
    return all_batch_ids, embeddings.numpy()  # 将 embeddings 转为 numpy 数组

def main():
    # 指定输入和输出文件路径
    train_fasta_file = "./Train/train_sequences.fasta"
    test_fasta_file = "./Test/testsuperset.fasta"  # 修改为您的测试集文件名
    train_embeddings_file = "train_embeddings.npy"
    train_ids_file = "train_ids.npy"
    test_embeddings_file = "test_embeddings.npy"
    test_ids_file = "test_ids.npy"

    # 检查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained("Bo1015/proteinglm-1b-mlm", trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(
        "Bo1015/proteinglm-1b-mlm",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用 float16 加速计算并减少显存占用
        output_hidden_states=True  # 添加此参数
    )
    model.to(device)

    # 加载序列
    print("正在加载训练序列...")
    train_sequences = load_sequences(train_fasta_file)
    print(f"加载了 {len(train_sequences)} 条训练序列。")
    
    print("正在加载测试序列...")
    test_sequences = load_sequences(test_fasta_file)
    print(f"加载了 {len(test_sequences)} 条测试序列。")

    # 生成训练集嵌入
    print("正在生成训练集嵌入...")
    train_ids, train_embeddings = generate_embeddings(train_sequences, tokenizer, model, device, batch_size=16)
    print("正在保存训练集嵌入和 IDs...")
    np.save(train_embeddings_file, train_embeddings)
    np.save(train_ids_file, np.array(train_ids))

    # 生成测试集嵌入
    print("正在生成测试集嵌入...")
    test_ids, test_embeddings = generate_embeddings(test_sequences, tokenizer, model, device, batch_size=16)
    print("正在保存测试集嵌入和 IDs...")
    np.save(test_embeddings_file, test_embeddings)
    np.save(test_ids_file, np.array(test_ids))

    print("全部嵌入已生成并保存。")

if __name__ == "__main__":
    main()
