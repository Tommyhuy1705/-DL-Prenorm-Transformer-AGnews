import argparse
import torch
from typing import Dict
import json
import os

from utils.dataset import get_dataloaders
from models.transformer_prenorm import TransformerPreNorm
from models.transformer_postnorm import TransformerPostNorm
from utils.metrics import accuracy, f1_macro, conf_matrix


def _infer_vocab_size(vocab_or_tokenizer):
    if hasattr(vocab_or_tokenizer, "vocab_size"):
        return int(vocab_or_tokenizer.vocab_size)
    if hasattr(vocab_or_tokenizer, "get_vocab"):
        return len(vocab_or_tokenizer.get_vocab())
    return len(vocab_or_tokenizer)


def load_model_from_checkpoint(checkpoint: str, model_type: str, device: torch.device, vocab_size: int):
    if model_type == "prenorm":
        model = TransformerPreNorm(vocab_size)
    else:
        model = TransformerPostNorm(vocab_size)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device)


def evaluate_checkpoint(
    checkpoint: str,
    model_type: str = "prenorm",
    device: str = None,
    batch_size: int = 32,
    seq_len: int = 128,
    max_vocab: int = 20000,
    use_hf_tokenizer: bool = False,
    tokenizer_name: str = "distilbert-base-uncased",
    test_subset: int = None,
    seed: int = 42,
) -> Dict[str, object]:
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    _, test_loader, vocab = get_dataloaders(
        batch_size=batch_size,
        seq_len=seq_len,
        max_vocab=max_vocab,
        use_hf_tokenizer=use_hf_tokenizer,
        tokenizer_name=tokenizer_name,
        train_subset=None,
        test_subset=test_subset,
        seed=seed,
    )
    vocab_size = _infer_vocab_size(vocab)
    model = load_model_from_checkpoint(checkpoint, model_type, device, vocab_size=vocab_size)
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, _mask, y = batch
            else:
                x, y = batch
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(p)
            targets.extend(y.tolist())

    results = {
        "accuracy": accuracy(preds, targets),
        "f1_macro": f1_macro(preds, targets),
        "confusion_matrix": conf_matrix(preds, targets),
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["prenorm", "postnorm"], default="prenorm")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--use-hf-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    res = evaluate_checkpoint(
        args.checkpoint,
        model_type=args.model,
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_vocab=args.max_vocab,
        use_hf_tokenizer=args.use_hf_tokenizer,
        tokenizer_name=args.tokenizer_name,
        test_subset=args.test_subset,
        seed=args.seed,
    )
    print("Độ chính xác:", res["accuracy"]) 
    print("F1 (macro):", res["f1_macro"]) 
    print("Ma trận nhầm lẫn:\n", res["confusion_matrix"]) 
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        serializable = {
            "accuracy": float(res["accuracy"]),
            "f1_macro": float(res["f1_macro"]),
            "confusion_matrix": res["confusion_matrix"].tolist(),
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print("Đã lưu kết quả JSON:", args.save_json)
