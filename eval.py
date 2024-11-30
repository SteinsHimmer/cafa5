import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
from load import loader


# 加载训练标签数据
train_terms = pd.read_csv("./Train/train_terms.tsv", sep="\t", header=None, names=["EntryID", "term", "aspect"])
print("train_terms.shape:", train_terms.shape)

from Bio import SeqIO

# 定义函数来加载 FASTA 文件
def load_fasta(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def generate_embeddings(sequences, tokenizer, model, batch_size=32):
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
            # 获取最后一层的隐藏状态
            last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_length, hidden_size)
            
            # 取平均池化作为序列的表示
            seq_embeddings = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            seq_embeddings = seq_embeddings.cpu().numpy()
            embeddings.append(seq_embeddings)
    
    embeddings = np.vstack(embeddings)
    return sequence_ids, embeddings

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5):
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_running_loss += loss.item() * inputs.size(0)
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

# 加载训练和测试序列
train_sequences = load_fasta("./Train/train_sequences.fasta")
test_sequences = load_fasta("./Test/test_sequences.fasta")

print("训练序列数量:", len(train_sequences))
print("测试序列数量:", len(test_sequences))

# 设置要选择的标签数量
num_of_labels = 1500

# 选择最频繁的 1500 个 GO 术语
labels = train_terms['term'].value_counts().index[:num_of_labels].tolist()
print("选择的标签数量:", len(labels))

# 过滤训练标签数据
train_terms_filtered = train_terms[train_terms['term'].isin(labels)]

# 创建蛋白质到 GO 术语的映射
protein_to_terms = train_terms_filtered.groupby('EntryID')['term'].apply(list).to_dict()

# 获取训练蛋白质 ID 列表
train_protein_ids = list(train_sequences.keys())

# 准备多标签二值矩阵
mlb = MultiLabelBinarizer(classes=labels)
train_labels = mlb.fit_transform([protein_to_terms.get(pid, []) for pid in train_protein_ids])

print("train_labels.shape:", train_labels.shape)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("Bo1015/proteinglm-1b-mlm", trust_remote_code=True, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(
    "Bo1015/proteinglm-1b-mlm",
    trust_remote_code=True,
    torch_dtype=torch.float16  # 使用 float16 可以加速计算并减少显存占用
)

# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 生成训练集嵌入
train_ids, train_embeddings = generate_embeddings(train_sequences, tokenizer, model, batch_size=16)
print("train_embeddings.shape:", train_embeddings.shape)

# 生成测试集嵌入
test_ids, test_embeddings = generate_embeddings(test_sequences, tokenizer, model, batch_size=16)
print("test_embeddings.shape:", test_embeddings.shape)

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_embeddings, train_labels, test_size=0.1, random_state=42)

# 创建数据集
train_dataset = ProteinDataset(X_train, y_train)
val_dataset = ProteinDataset(X_val, y_val)

# 创建数据加载器
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 初始化模型
input_size = train_embeddings.shape[1]
num_labels = len(labels)

classifier = MultiLabelClassifier(input_size, num_labels).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# 创建测试数据集
test_dataset = ProteinDataset(test_embeddings)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

classifier.eval()
all_predictions = []

with torch.no_grad():
    for inputs in tqdm(test_loader, desc="Predicting on Test Data"):
        inputs = inputs.to(device)
        outputs = classifier(inputs)
        all_predictions.append(outputs.cpu().numpy())

# 将所有预测结果拼接起来
all_predictions = np.vstack(all_predictions)
print("all_predictions.shape:", all_predictions.shape)