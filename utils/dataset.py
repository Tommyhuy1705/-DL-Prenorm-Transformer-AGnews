from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class AGNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: dict, seq_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def _encode(self, text: str):
        # Token hoá đơn giản theo khoảng trắng và ánh xạ sang id
        tokens = text.split()
        ids = [self.vocab.get(t, self.vocab.get("<unk>")) for t in tokens][: self.seq_len]
        if len(ids) < self.seq_len:
            ids = ids + [self.vocab.get("<pad>")] * (self.seq_len - len(ids))
        return ids

    def __getitem__(self, idx):
        return torch.tensor(self._encode(self.texts[idx]), dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class HFAGNewsDataset(Dataset):

    def __init__(self, texts: List[str], labels: List[int], tokenizer, seq_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.seq_len, return_tensors='pt')
        # enc fields are tensors with shape (1, seq_len)
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        return input_ids, attention_mask, torch.tensor(self.labels[idx], dtype=torch.long)


def build_simple_vocab(texts: List[str], max_size: int = 20000) -> dict:
    freq = {}
    for t in texts:
        for w in t.split():
            freq[w] = freq.get(w, 0) + 1
    vocab_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_size]
    stoi = {w: i + 4 for i, (w, _) in enumerate(vocab_items)}
    # reserved tokens
    stoi["<pad>"] = 0
    stoi["<unk>"] = 1
    stoi["<cls>"] = 2
    stoi["<sep>"] = 3
    return stoi


def load_ag_news(split: str = "train") -> Tuple[List[str], List[int]]:
    ds = load_dataset("ag_news", split=split)
    texts = [" ".join(x["text"].split()) for x in ds]
    labels = [int(x["label"]) for x in ds]
    return texts, labels


def _subsample(texts: List[str], labels: List[int], n_samples: Optional[int], seed: int) -> Tuple[List[str], List[int]]:
    if n_samples is None or n_samples <= 0 or n_samples >= len(texts):
        return texts, labels
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(texts), generator=g)[:n_samples].tolist()
    new_texts = [texts[i] for i in indices]
    new_labels = [labels[i] for i in indices]
    return new_texts, new_labels


def get_dataloaders(
    batch_size: int = 32,
    seq_len: int = 128,
    max_vocab: int = 20000,
    use_hf_tokenizer: bool = False,
    tokenizer_name: str = "distilbert-base-uncased",
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    seed: int = 42,
):

    train_texts, train_labels = load_ag_news("train")
    test_texts, test_labels = load_ag_news("test")

    # Cho phép chạy nhanh bằng tập con để debug/thử nghiệm nhanh.
    train_texts, train_labels = _subsample(train_texts, train_labels, train_subset, seed)
    test_texts, test_labels = _subsample(test_texts, test_labels, test_subset, seed)

    try:
        # Nếu có AutoTokenizer và được bật, dùng tokenizer Hugging Face.
        if AutoTokenizer is not None and use_hf_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            train_ds = HFAGNewsDataset(train_texts, train_labels, tokenizer, seq_len)
            test_ds = HFAGNewsDataset(test_texts, test_labels, tokenizer, seq_len)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size)
            return train_loader, test_loader, tokenizer
    except Exception:
        # Nếu lỗi khi tải tokenizer, fallback về simple vocab.
        pass

    vocab = build_simple_vocab(train_texts, max_size=max_vocab)
    train_ds = AGNewsDataset(train_texts, train_labels, vocab, seq_len)
    test_ds = AGNewsDataset(test_texts, test_labels, vocab, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader, vocab