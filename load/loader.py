import torch.nn as nn
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return embedding, label
        else:
            return embedding
        
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, num_labels):
        super(MultiLabelClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)