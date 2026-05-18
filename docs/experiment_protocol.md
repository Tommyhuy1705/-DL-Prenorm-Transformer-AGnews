# Quy trình thực nghiệm chuẩn (Pre-Norm vs Post-Norm)

## Mục tiêu

So sánh công bằng hai kiến trúc `prenorm` và `postnorm` trên AG News, chỉ thay đổi vị trí LayerNorm, giữ nguyên các yếu tố khác.

## Thiết lập bắt buộc để công bằng

- Cùng seed (ví dụ: 42)
- Cùng tokenizer (simple hoặc Hugging Face)
- Cùng hyperparameter: `epochs`, `batch_size`, `lr`, `seq_len`, `max_vocab`
- Cùng tập dữ liệu train/test (hoặc cùng subset khi debug)

## Lệnh chạy gợi ý

```bash
python experiments.py --epochs 3 --batch-size 64 --lr 1e-3 --seed 42
```

Khi cần chạy nhanh để debug CPU:

```bash
python experiments.py --epochs 1 --batch-size 32 --train-subset 5000 --test-subset 1000 --seed 42
```

## Kết quả cần thu thập

- Theo epoch: `train_loss`, `val_acc` (lưu ở `logs/history_*.csv`)
- Cuối cùng: `test_accuracy`, `test_f1_macro`, `confusion_matrix`

## Đầu ra đã được lưu tự động

- `results/comparison_results.csv`: bảng so sánh cuối cùng
- `results/confusion_matrices.json`: ma trận nhầm lẫn của hai mô hình

## Diễn giải kết quả trong báo cáo

- So sánh tốc độ hội tụ: nhìn đường `train_loss`, `val_acc` theo epoch.
- So sánh hiệu năng tổng thể: `test_accuracy`, `test_f1_macro`.
- Phân tích lỗi theo lớp: dựa trên `confusion_matrix`.
- Kết luận điểm mạnh/yếu giữa Pre-Norm và Post-Norm trong bối cảnh text classification.
