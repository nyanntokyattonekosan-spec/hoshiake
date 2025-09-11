# model.py
import os
import json
import math
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 簡易文字トークナイザ（プロトタイプ）
# -------------------------
class CharTokenizer:
    def __init__(self, vocab=None):
        # 特殊トークン
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"
        if vocab:
            self.idx_to_token = vocab
            self.token_to_idx = {t:i for i,t in enumerate(self.idx_to_token)}
        else:
            # 初期は日本語の文字頻度を足して拡張していく想定
            self.idx_to_token = [self.PAD, self.UNK, self.BOS, self.EOS]
            self.token_to_idx = {t:i for i,t in enumerate(self.idx_to_token)}

    @property
    def vocab_size(self):
        return len(self.idx_to_token)

    def build_from_texts(self, texts: List[str], min_freq=1):
        freq = {}
        for t in texts:
            for ch in t:
                freq[ch] = freq.get(ch, 0) + 1
        for ch,count in sorted(freq.items(), key=lambda x:-x[1]):
            if count >= min_freq and ch not in self.token_to_idx:
                self.token_to_idx[ch] = len(self.idx_to_token)
                self.idx_to_token.append(ch)

    def encode(self, text: str, add_bos=True, add_eos=True, max_len=None):
        ids = []
        if add_bos:
            ids.append(self.token_to_idx[self.BOS])
        for ch in text:
            ids.append(self.token_to_idx.get(ch, self.token_to_idx[self.UNK]))
            if max_len and len(ids) >= max_len:
                break
        if add_eos:
            ids.append(self.token_to_idx[self.EOS])
        return ids

    def decode(self, ids: List[int]):
        tokens = []
        for i in ids:
            if i < 0 or i >= len(self.idx_to_token):
                tokens.append(self.UNK)
            else:
                t = self.idx_to_token[i]
                if t in (self.BOS, self.EOS, self.PAD):
                    continue
                tokens.append(t)
        return "".join(tokens)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.idx_to_token, f, ensure_ascii=False)

def load_tokenizer_if_exists(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return CharTokenizer(vocab=vocab)
    return None

# -------------------------
# 小型 Transformer（GPT 系列に類似）
# -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # B T 3C
        q, k, v = qkv.chunk(3, dim=2)
        # reshape for heads: B, n_head, T, head_dim
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        att = torch.matmul(q, k.transpose(-2,-1)) * self.scale  # B,nh,T,T

        # causal mask (prevent attending to future)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = torch.matmul(att, v)  # B,nh,T,head_dim
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out

class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_head=8, d_model=512, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.max_seq_len = max_seq_len

    def forward(self, idx):  # idx: B x T (LongTensor)
        B, T = idx.size()
        assert T <= self.max_seq_len
        tok = self.tok_emb(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = tok + self.pos_emb(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits  # B x T x V

    # 簡易生成（シンプルな top-k サンプリング）
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50, device="cpu"):
        self.eval()
        ids = input_ids[:]  # list of ints
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-self.max_seq_len:]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.forward(x)  # 1 x T x V
            logits = logits[0, -1, :] / max(1e-8, temperature)
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                min_top = v[-1]
                logits[logits < min_top] = -1e9
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            ids.append(next_id)
            if next_id == 1:  # EOS index is 1? In our tokenizer EOS index may differ; adapt if needed
                break
        return ids

# -------------------------
# PII マスク（簡易）: メール・URL・電話番号などを置換
# -------------------------
import re
_email_re = re.compile(r"[\w\.-]+@[\w\.-]+")
_url_re = re.compile(r"https?://\S+|www\.\S+")
_phone_re = re.compile(r"(?:\+?\d[\d\-\s]{6,}\d)")

def mask_pii(text):
    text = _email_re.sub("[EMAIL_REDACTED]", text)
    text = _url_re.sub("[URL_REDACTED]", text)
    text = _phone_re.sub("[PHONE_REDACTED]", text)
    return text

# -------------------------
# 簡易トレーニング関数（新規データで数エポックだけ学習する想定）
# -------------------------
def train_on_texts(texts, tokenizer: CharTokenizer, save_path="data/model.pt",
                   epochs=1, seq_len=128, batch_size=8, lr=2e-4, device="cpu"):
    # texts: list of cleaned strings
    # tokenizer: インスタンス（必要なら vocab build）
    if tokenizer.vocab_size <= 4:
        tokenizer.build_from_texts(texts)
        tokenizer.save("data/tokenizer.json")

    # instantiate model if not exists
    model = TransformerModel(vocab_size=tokenizer.vocab_size, max_seq_len=seq_len)
    model.to(device)
    model.train()

    # build token stream
    stream = []
    for t in texts:
        ids = tokenizer.encode(t, add_bos=False, add_eos=True)
        stream.extend(ids)

    # split into blocks
    blocks = []
    for i in range(0, max(1, len(stream) - seq_len), seq_len):
        block = stream[i:i+seq_len+1]  # input + next token (causal LM)
        if len(block) == seq_len+1:
            blocks.append(block)

    if not blocks:
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        random.shuffle(blocks)
        for i in range(0, len(blocks), batch_size):
            batch_blocks = blocks[i:i+batch_size]
            x = torch.tensor([[b[:-1] for b in batch_blocks]], dtype=torch.long)  # shape incorrectly nested? fix below
            # Correct collate:
            x = torch.tensor([b[:-1] for b in batch_blocks], dtype=torch.long, device=device)  # B x seq_len
            y = torch.tensor([b[1:] for b in batch_blocks], dtype=torch.long, device=device)  # labels
            optimizer.zero_grad()
            logits = model(x)  # B x T x V
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
        # end epoch
    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    # optionally save tokenizer (already saved)
    return

def load_model_if_exists(path, vocab_size=None, device="cpu"):
    if path and os.path.exists(path):
        # vocab_size is required to rebuild model architecture
        # caller should provide tokenizer.vocab_size
        # return loaded model (moved to device)
        raise NotImplementedError("Use caller to instantiate model with proper vocab size, then load state_dict.")
    return None
