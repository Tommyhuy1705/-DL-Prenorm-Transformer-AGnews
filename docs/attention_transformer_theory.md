# Attention và Transformer cho Text Classification

## 1) Module Attention: kiến trúc cốt lõi

Với đầu vào đã được chiếu tuyến tính thành ba ma trận:

- Query: $Q$
- Key: $K$
- Value: $V$

Scaled Dot-Product Attention được tính theo công thức:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Ý nghĩa:

- Tích $QK^T$ đo độ tương quan giữa token truy vấn và token ngữ cảnh.
- Chia cho $\sqrt{d_k}$ để ổn định gradient khi kích thước ẩn lớn.
- Softmax chuẩn hóa trọng số attention theo từng token.
- Nhân với $V$ để tổng hợp thông tin ngữ cảnh có trọng số.

## 2) Phân loại các kỹ thuật Attention

### 2.1 Self-Attention

- $Q, K, V$ lấy từ cùng một chuỗi đầu vào.
- Mục tiêu: học phụ thuộc ngữ cảnh nội bộ trong câu/tài liệu.
- Ứng dụng chính: Encoder Transformer cho phân loại văn bản.

### 2.2 Cross-Attention

- $Q$ lấy từ chuỗi A, còn $K, V$ lấy từ chuỗi B.
- Mục tiêu: căn chỉnh thông tin giữa hai nguồn khác nhau.
- Ứng dụng chính: Decoder trong mô hình Encoder-Decoder (dịch máy, tóm tắt có điều kiện).

### 2.3 Multi-Head Attention

- Chia không gian ẩn thành nhiều head và chạy attention song song.
- Mục tiêu: cho phép mô hình học nhiều kiểu quan hệ ngữ nghĩa ở các không gian con khác nhau.
- Lưu ý: Multi-Head là cách triển khai, có thể áp dụng cho Self-Attention hoặc Cross-Attention.

## 3) So sánh mục đích/chức năng

- Self-Attention: nắm bắt phụ thuộc nội bộ trong cùng chuỗi.
- Cross-Attention: trao đổi thông tin giữa hai chuỗi khác nhau.
- Multi-Head: tăng năng lực biểu diễn nhờ nhiều góc nhìn attention cùng lúc.

## 4) Transformer tích hợp Attention

Mỗi lớp encoder Transformer gồm:

1. Multi-Head Self-Attention
2. Feed-Forward Network (FFN)
3. Residual connection
4. Layer Normalization (vị trí khác nhau giữa Post-Norm và Pre-Norm)

## 5) Liên hệ với code hiện tại

- Module attention nằm ở `models/attention.py`:
  - `ScaledDotProductAttention`
  - `MultiHeadAttention`
- Hai kiến trúc so sánh:
  - `models/transformer_postnorm.py`
  - `models/transformer_prenorm.py`

Trong project này, cả hai mô hình đều dùng cùng module attention; khác nhau ở vị trí LayerNorm:

- Post-Norm: Add & Residual rồi mới LayerNorm.
- Pre-Norm: LayerNorm trước sub-layer rồi Add & Residual.

## 6) Cài đặt cho bài toán Text Classification

Pipeline thực nghiệm:

1. Tokenize văn bản (simple tokenizer hoặc Hugging Face tokenizer).
2. Chuyển thành `input_ids` và padding/truncation theo `seq_len`.
3. Đưa qua Transformer encoder nhiều lớp.
4. Pooling theo chuỗi (trong code hiện tại dùng `AdaptiveAvgPool1d`).
5. Linear classifier dự đoán 4 lớp AG News.

## 7) Trọng tâm Pre-Norm cho thực nghiệm

Pre-Norm thường ổn định hơn khi tăng độ sâu vì gradient lan truyền thuận lợi hơn qua các residual path. Trong báo cáo nên đánh giá:

- tốc độ hội tụ (loss/val acc theo epoch)
- hiệu năng cuối (accuracy/F1)
- ma trận nhầm lẫn theo từng lớp
