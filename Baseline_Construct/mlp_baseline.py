# MLP Baseline for RNA k-mer translation
# Predicts first token of target sequence based on average k-mer embedding

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import random

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("translation_pairs_k7.tsv", sep="\t", header=None, names=["source", "target"])
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
train_df, test_df = df[:split], df[split:]

# -------------------------
# Vocab and tokenization
# -------------------------
def tokenize(seq): return seq.strip().split()

def build_vocab(seqs):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in seqs:
        for token in tokenize(seq):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

src_vocab = build_vocab(train_df["source"])
tgt_vocab = build_vocab(train_df["target"])

def encode(seq, vocab, max_len):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(seq)]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

# -------------------------
# Dataset
# -------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab, max_len=20):
        self.samples = [
            (encode(row["source"], src_vocab, max_len), encode(row["target"], tgt_vocab, max_len))
            for _, row in df.iterrows()
        ]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        return torch.tensor(src), torch.tensor(tgt)

train_dl = DataLoader(EmbeddingDataset(train_df, src_vocab, tgt_vocab), batch_size=32, shuffle=True)
test_dl = DataLoader(EmbeddingDataset(test_df, src_vocab, tgt_vocab), batch_size=32)

# -------------------------
# MLP Model
# -------------------------
class MLPBaseline(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        return self.mlp(x)

# -------------------------
# Training
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPBaseline(len(src_vocab), emb_dim=128, hidden_dim=256, out_dim=len(tgt_vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for src, tgt in train_dl:
        src, tgt = src.to(device), tgt[:, 0].to(device)
        optimizer.zero_grad()
        loss = criterion(model(src), tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Loss: {total_loss / len(train_dl):.4f}")

# -------------------------
# Evaluation
# -------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for src, tgt in test_dl:
        src, tgt = src.to(device), tgt[:, 0].to(device)
        preds = model(src).argmax(dim=1)
        correct += (preds == tgt).sum().item()
        total += tgt.size(0)
print(f"Token-level Accuracy (first target token): {correct / total:.4f}")
