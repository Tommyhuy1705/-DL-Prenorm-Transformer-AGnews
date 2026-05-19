# Pre-Norm vs Post-Norm Transformer cho Phân loại Văn bản

**Nhóm:** Group 2
## Thành viên nhóm

| Name | Student ID | Role |
|------|------------|------|
| Trần Viết Gia Huy | 31231027056 |  |
| Nguyễn Minh Nhựt | 31231022656 |  |
| Nguyễn Trọng Hưởng | 31231023691 |  |
| Nguyễn Quốc Khánh | 31231021198 |  |
| Ngô Chánh Phong | 31231021197 |  |
| Tô Xuân Đông | 31231025345 |  |
| Dương Quang Đông | 31231020389 |  |
| Nguyễn Đình Lương | 31231027411 |  |
| Võ Nguyên Bảo | 31231021638  |  |
| Trần Thành Đạt | 31231021353  |  |
## Tổng quan dự án

Dự án này triển khai và so sánh hai biến thể bộ mã hóa Transformer cho bài toán phân loại văn bản trên tập dữ liệu AG News:
- Transformer Post-Norm (chuẩn): Layer Normalization được áp dụng sau khi cộng kết nối tắt (residual).
- Transformer Pre-Norm: Layer Normalization được áp dụng trước các khối attention và feed-forward.
Mục tiêu là phân tích sự khác biệt về ổn định huấn luyện, tốc độ hội tụ và hiệu năng cuối cùng giữa hai chiến lược normalization này.

## Dataset
Sử dụng tập AG News, gồm 4 lớp (World, Sports, Business, Sci/Tech).

## Kiến trúc
- Post-Norm: Residual -> Add -> LayerNorm (LayerNorm áp sau khi cộng residual).
- Pre-Norm: LayerNorm -> Sub-layer -> Add (LayerNorm áp trước Attention/FFN).
Sự khác nhau này ảnh hưởng đến luồng gradient và độ ổn định khi huấn luyện; Pre-Norm thường giúp ổn định khi mạng sâu hơn.

## Cấu trúc thư mục
```
Prenorm-Transformer-AGnews/
├─ data/
│  ├─ raw/ (.gitkeep)
│  └─ processed/ (.gitkeep)
├─ models/
│  ├─ __init__.py
│  ├─ attention.py
│  ├─ transformer_postnorm.py
│  └─ transformer_prenorm.py
├─ utils/
│  ├─ __init__.py
│  ├─ dataset.py
│  └─ metrics.py
├─ notebooks/
│  ├─ 01_EDA_AGNews.ipynb
│  └─ 02_Result_Visualization.ipynb
│  └─ 03_Train_and_Test.ipynb
├─ train.py
├─ evaluate.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Cài đặt
Clone repository và cài phụ thuộc:

```bash
git clone <repo_url>
cd Prenorm-Transformer-AGnews
pip install -r requirements.txt
```

## Sử dụng
Ví dụ huấn luyện:

```bash
python train.py --model prenorm --epochs 10 --batch-size 64
```

Ví dụ đánh giá checkpoint đã lưu:
```bash
python evaluate.py --model prenorm --checkpoint checkpoints/prenorm_<timestamp>.pt
```

Chạy thực nghiệm so sánh công bằng Pre-Norm vs Post-Norm và lưu bảng kết quả:

```bash
python experiments.py --epochs 3 --batch-size 64 --lr 1e-3 --seed 42
```

File kết quả được lưu tại:

- `results/comparison_results.csv`
- `results/confusion_matrices.json`

## Ghi chú
- Đây là khung mã nguồn sạch, mô-đun để thực hiện thí nghiệm.
- Nên mở rộng với logging, scheduler, mixed precision và tìm kiếm siêu tham số cho báo cáo cuối cùng.
# Pre-Norm vs Post-Norm Transformer for Text Classification

**Group:** Group 2

## Group Members

| Name | Student ID | Role |
|------|------------|------|
| Trần Viết Gia Huy | 31231027056 |  |
| Nguyễn Minh Nhựt | 31231022656 |  |
| Nguyễn Trọng Hưởng | 31231023691 |  |
| Nguyễn Quốc Khánh | 31231021198 |  |
| Ngô Chánh Phong | 31231021197 |  |
| Tô Xuân Đông | 31231025345 |  |
| Dương Quang Đông | 31231020389 |  |
| Nguyễn Đình Lương | 31231027411 |  |
| Võ Nguyên Bảo | 31231021638  |  |
| Trần Thành Đạt | 31231021353  |  |

## Project Overview

This project implements and compares two Transformer encoder variants for text classification on the AG News dataset:

- Post-Norm Transformer (standard): Layer normalization is applied after the residual connection.
- Pre-Norm Transformer: Layer normalization is applied before the attention and feed-forward sublayers.

The goal is to study stability, convergence behavior, and final performance differences between the two normalization strategies.

## Dataset

We use the AG News dataset, a 4-class news classification dataset (labels: World, Sports, Business, Sci/Tech).

## Architecture

- Post-Norm: Residual -> Add -> LayerNorm (i.e., LayerNorm applied after adding residual).
- Pre-Norm: LayerNorm -> Sub-layer -> Add (i.e., LayerNorm applied before attention/FFN blocks).

These choices affect gradient flow and training stability; Pre-Norm often stabilizes deeper Transformer stacks.

## Directory Structure

```
Prenorm-Transformer-AGnews/
├─ data/
│  ├─ raw/ (.gitkeep)
│  └─ processed/ (.gitkeep)
├─ models/
│  ├─ __init__.py
│  ├─ attention.py
│  ├─ transformer_postnorm.py
│  └─ transformer_prenorm.py
├─ utils/
│  ├─ __init__.py
│  ├─ dataset.py
│  └─ metrics.py
├─ notebooks/
│  ├─ 01_EDA_AGNews.ipynb
│  └─ 02_Result_Visualization.ipynb
├─ train.py
├─ evaluate.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Setup & Installation

Clone the repository and install dependencies:

```bash
git clone <repo_url>
cd Prenorm-Transformer-AGnews
pip install -r requirements.txt
```

## Usage

Train a model (example):

```bash
python train.py --model prenorm --epochs 10 --batch-size 64
```

Evaluate a saved checkpoint:

```bash
python evaluate.py --model prenorm --checkpoint checkpoints/prenorm_<timestamp>.pt
```

## Notes

- The provided code is a clean, modular starting point for experimentation.
- Extend training with logging, LR schedulers, mixed precision, and hyperparameter sweeps for the final report.
