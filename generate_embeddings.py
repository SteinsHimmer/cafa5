# generate_embeddings.py

import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from Bio import SeqIO

def load_sequences(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def generate_embeddings(sequences, tokenizer, model, device, batch_size=16):
    model.eval()
    embeddings = []
    sequence_ids = list(sequences.keys())
    sequence_list = list(sequences.values())

    with torch.no_grad():
        for i in tqdm(range(0, len(sequence_list), batch_size), desc="Generating Embeddings"):
            batch_seqs = sequence_list[i:i+batch_size]
            encoded = tokenizer(batch_seqs, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_length, hidden_size)

            # 平均池化获取序列嵌入
            seq_embeddings = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            seq_embeddings = seq_embeddings.cpu().numpy()
            embeddings.append(seq_embeddings)

    embeddings = np.vstack(embeddings)
    return sequence_ids, embeddings

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
    model = AutoModelForMaskedLM.from_pretrained(
        "Bo1015/proteinglm-1b-mlm",
        trust_remote_code=True,
        torch_dtype=torch.float16  # 使用 float16 加速计算并减少显存占用
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