# Pre-Norm vs Post-Norm Transformer cho Phân loại Văn bản

**Nhóm:** Group 2
## Thành viên nhóm

| Name | Student ID | Role |
|------|------------|------|
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
| Dương Quang Đông | 31231020389 | Báo cáo: 3.3 |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

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
