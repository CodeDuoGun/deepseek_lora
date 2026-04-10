# file: word2vec_pytorch_sgns.py
# 依赖: torch
# pip install torch --extra-index-url https://download.pytorch.org/whl/cu118   (或常规pip安装cpu版)
import math
import random
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 1. 简单的语料与预处理
# -------------------------
corpus = [
    "we are what we repeatedly do",
    "excellence then is not an act but a habit",
    "we are what we do repeatedly",
    "practice makes perfect",
    "you are what you repeatedly do"
]

# 小写并分词（真实应用需要更完善的分词/清洗）
tokens = []
for line in corpus:
    tokens += line.lower().split()

# 建词表
min_count = 1
counter = Counter(tokens)
print(counter, "total unique words:", len(counter))
vocab = [w for w, c in counter.items() if c >= min_count]
vocab_size = len(vocab)
idx2word = vocab
word2idx = {w: i for i, w in enumerate(idx2word)}

print("vocab size:", vocab_size)
# -------------------------
# 2. 生成训练样本 (skip-gram)
# -------------------------
def generate_skipgram_pairs(tokens: List[str], word2idx: dict, window_size=2) -> List[Tuple[int,int]]:
    pairs = []
    for i, w in enumerate(tokens):
        w_idx = word2idx[w]
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if j == i:
                continue
            context_word = tokens[j]
            pairs.append((w_idx, word2idx[context_word]))  # (center, context)
    return pairs

pairs = generate_skipgram_pairs(tokens, word2idx, window_size=2)
print("sample pairs (center->context):", pairs[:10])

# -------------------------
# 3. 负采样分布（Unigram^0.75）
# -------------------------
def unigram_distribution(counter: Counter, word2idx: dict, power=0.75):
    freq = torch.tensor([counter[w] for w in idx2word], dtype=torch.float)
    freq = freq.pow(power)
    probs = freq / freq.sum()
    return probs

unigram_probs = unigram_distribution(counter, word2idx, power=0.75)
# -------------------------
# 4. Dataset 与 DataLoader
# -------------------------
class SkipGramDataset(Dataset):
    def __init__(self, pairs, unigram_probs, neg_sample_num=5):
        self.pairs = pairs
        self.unigram_probs = unigram_probs
        self.neg_sample_num = neg_sample_num
        # precompute alias? 这里使用 torch.multinomial 简单采样

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        # 采样负样本（避免采到正样本，可接受小概率）
        negs = torch.multinomial(self.unigram_probs, self.neg_sample_num, replacement=True).long()
        return torch.LongTensor([center]), torch.LongTensor([context]), negs

# -------------------------
# 5. 模型定义（Embedding + Negative Sampling Loss）
# -------------------------
class SGNS(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.in_embed = nn.Embedding(vocab_size, emb_dim)   # center word emb
        self.out_embed = nn.Embedding(vocab_size, emb_dim)  # context word emb
        self.init_emb()

    def init_emb(self):
        # 通常用均匀或正态初始化
        initrange = 0.5 / self.emb_dim
        nn.init.uniform_(self.in_embed.weight, -initrange, initrange)
        nn.init.uniform_(self.out_embed.weight, -0, 0)

    def forward(self, centers, contexts, negs):
        # centers: (B,1)  contexts: (B,1)  negs: (B,k)
        v_c = self.in_embed(centers).squeeze(1)    # (B, D)
        u_o = self.out_embed(contexts).squeeze(1)  # (B, D)
        negs_emb = self.out_embed(negs)            # (B, k, D)

        # positive score: sigmoid(u_o · v_c)
        pos_score = torch.sum(u_o * v_c, dim=1)   # (B,)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()

        # negative score: sigmoid(-u_k · v_c)
        neg_score = torch.bmm(negs_emb.neg(), v_c.unsqueeze(2)).squeeze(2)  # (B, k)
        neg_loss = -torch.log(torch.sigmoid(neg_score) + 1e-10).sum(1).mean()

        loss = pos_loss + neg_loss
        return loss

    def get_embedding(self):
        # 返回训练好的输入 embedding（常用）
        return self.in_embed.weight.data.cpu().numpy()

# -------------------------
# 6. 训练
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
emb_dim = 50
batch_size = 16
neg_samples = 5
dataset = SkipGramDataset(pairs, unigram_probs, neg_sample_num=neg_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SGNS(vocab_size, emb_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

epochs = 1000
for epoch in range(epochs):
    total_loss = 0.0
    for centers, contexts, negs in dataloader:
        centers = centers.to(device)
        contexts = contexts.to(device)
        negs = negs.to(device)
        loss = model(centers, contexts, negs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * centers.size(0)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(dataset):.4f}")

# -------------------------
# 7. 使用 & 简单相似度测试
# -------------------------
emb = model.get_embedding()  # numpy array (vocab_size, emb_dim)
import numpy as np
def most_similar(word, topk=5):
    if word not in word2idx:
        return []
    v = emb[word2idx[word]]
    sims = np.dot(emb, v) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(v) + 1e-8)
    idxs = np.argsort(-sims)[:topk+1]
    return [(idx2word[i], float(sims[i])) for i in idxs if idx2word[i] != word][:topk]

print("Most similar to 'we':", most_similar("we"))
print("Most similar to 'what':", most_similar("what"))
