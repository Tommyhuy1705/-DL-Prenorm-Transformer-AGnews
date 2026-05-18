"""Hàm tiện ích huấn luyện cho Transformer Pre-Norm và Post-Norm.

Module này cung cấp các hàm để xây model, huấn luyện theo epoch,
đánh giá theo epoch, và hỗ trợ thực nghiệm so sánh công bằng.
"""

import argparse
import csv
import os
import random
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.dataset import get_dataloaders
from models.transformer_prenorm import TransformerPreNorm
from models.transformer_postnorm import TransformerPostNorm
from utils.metrics import accuracy


def build_model(model_type: str, vocab_size: int, device: torch.device, **kwargs) -> torch.nn.Module:
    """Xây và trả về model Transformer trên `device`."""
    if model_type == "prenorm":
        model = TransformerPreNorm(vocab_size, **kwargs)
    else:
        model = TransformerPostNorm(vocab_size, **kwargs)
    return model.to(device)


def train_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        if len(batch) == 3:
            x, _mask, y = batch
        else:
            x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, list, list]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, _mask, y = batch
            else:
                x, y = batch
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(p)
            targets.extend(y.tolist())
    return accuracy(preds, targets), preds, targets


def set_seed(seed: int = 42) -> None:
    """Cố định seed để thực nghiệm có tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_vocab_size(vocab_or_tokenizer) -> int:
    """Suy ra vocab_size từ dict vocab hoặc tokenizer object."""
    if hasattr(vocab_or_tokenizer, "vocab_size"):
        return int(vocab_or_tokenizer.vocab_size)
    if hasattr(vocab_or_tokenizer, "get_vocab"):
        return len(vocab_or_tokenizer.get_vocab())
    return len(vocab_or_tokenizer)


def train_loop(
    model_type: str,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = None,
    save_dir: str = "checkpoints",
    log_dir: str = "logs",
    seed: int = 42,
    seq_len: int = 128,
    max_vocab: int = 20000,
    use_hf_tokenizer: bool = False,
    tokenizer_name: str = "distilbert-base-uncased",
    train_subset: int = None,
    test_subset: int = None,
    early_stop_patience: int = None,
    early_stop_min_delta: float = 0.0,
) -> str:
    """Vòng huấn luyện mức cao; trả về đường dẫn checkpoint.

    Hàm này in `loss` và `val acc` theo từng epoch, đồng thời lưu lịch sử
    vào file CSV để tiện vẽ biểu đồ/báo cáo.
    """
    set_seed(seed)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader, vocab = get_dataloaders(
        batch_size=batch_size,
        seq_len=seq_len,
        max_vocab=max_vocab,
        use_hf_tokenizer=use_hf_tokenizer,
        tokenizer_name=tokenizer_name,
        train_subset=train_subset,
        test_subset=test_subset,
        seed=seed,
    )
    vocab_size = infer_vocab_size(vocab)
    model = build_model(model_type, vocab_size, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    history_rows = []
    best_acc = float("-inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc, _, _ = evaluate_model(model, test_loader, device)
        history_rows.append({"epoch": epoch, "train_loss": loss, "val_acc": acc})
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Độ chính xác (val): {acc:.4f}")

        if acc > best_acc + early_stop_min_delta:
            best_acc = acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif early_stop_patience is not None:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(
                    f"Early stopping tại epoch {epoch} (best val acc = {best_acc:.4f} ở epoch {best_epoch})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ts = int(time.time())
    ckpt_path = os.path.join(save_dir, f"{model_type}_{ts}.pt")
    torch.save(model.state_dict(), ckpt_path)
    history_path = os.path.join(log_dir, f"history_{model_type}_{ts}.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history_rows)

    print("Đã lưu checkpoint:", ckpt_path)
    print("Đã lưu log huấn luyện:", history_path)
    return ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Transformer")
    parser.add_argument("--model", choices=["prenorm", "postnorm"], default="prenorm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--use-hf-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        seq_len=args.seq_len,
        max_vocab=args.max_vocab,
        use_hf_tokenizer=args.use_hf_tokenizer,
        tokenizer_name=args.tokenizer_name,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
