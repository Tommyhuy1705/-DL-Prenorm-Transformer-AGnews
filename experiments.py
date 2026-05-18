import argparse
import csv
import glob
import json
import os
from typing import Dict

from train import train_loop
from evaluate import evaluate_checkpoint


def _read_last_row(csv_path: str) -> Dict[str, float]:
    last = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    if last is None:
        return {"epoch": None, "train_loss": None, "val_acc": None}
    return {
        "epoch": int(last["epoch"]),
        "train_loss": float(last["train_loss"]),
        "val_acc": float(last["val_acc"]),
    }


def _latest_history_file(model_type: str, log_dir: str = "logs") -> str:
    pattern = os.path.join(log_dir, f"history_{model_type}_*.csv")
    files = glob.glob(pattern)
    if not files:
        return ""
    files.sort(key=os.path.getmtime)
    return files[-1]


def run_one_model(model_type: str, args) -> Dict[str, object]:
    ckpt = train_loop(
        model_type=model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        seq_len=args.seq_len,
        max_vocab=args.max_vocab,
        use_hf_tokenizer=args.use_hf_tokenizer,
        tokenizer_name=args.tokenizer_name,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    eval_res = evaluate_checkpoint(
        checkpoint=ckpt,
        model_type=model_type,
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_vocab=args.max_vocab,
        use_hf_tokenizer=args.use_hf_tokenizer,
        tokenizer_name=args.tokenizer_name,
        test_subset=args.test_subset,
        seed=args.seed,
    )

    history_path = _latest_history_file(model_type, args.log_dir)
    history_last = _read_last_row(history_path) if history_path else {"epoch": None, "train_loss": None, "val_acc": None}

    return {
        "model": model_type,
        "checkpoint": ckpt,
        "history_file": history_path,
        "last_epoch": history_last["epoch"],
        "train_loss": history_last["train_loss"],
        "val_acc": history_last["val_acc"],
        "test_accuracy": float(eval_res["accuracy"]),
        "test_f1_macro": float(eval_res["f1_macro"]),
        "confusion_matrix": eval_res["confusion_matrix"].tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy so sánh Pre-Norm vs Post-Norm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--use-hf-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    prenorm = run_one_model("prenorm", args)
    postnorm = run_one_model("postnorm", args)

    table_path = os.path.join(args.result_dir, "comparison_results.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "checkpoint",
                "history_file",
                "last_epoch",
                "train_loss",
                "val_acc",
                "test_accuracy",
                "test_f1_macro",
            ],
        )
        writer.writeheader()
        writer.writerow({k: v for k, v in prenorm.items() if k != "confusion_matrix"})
        writer.writerow({k: v for k, v in postnorm.items() if k != "confusion_matrix"})

    cm_path = os.path.join(args.result_dir, "confusion_matrices.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prenorm": prenorm["confusion_matrix"],
                "postnorm": postnorm["confusion_matrix"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Đã lưu bảng so sánh:", table_path)
    print("Đã lưu confusion matrices:", cm_path)


if __name__ == "__main__":
    main()
