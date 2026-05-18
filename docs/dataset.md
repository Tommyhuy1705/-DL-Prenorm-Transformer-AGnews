# Mô tả chi tiết Dataset — AG News

## Tổng quan

AG News là một tập dữ liệu tiếng Anh dùng cho bài toán phân loại tin tức (news classification). Dataset được xây dựng từ nguồn tin tức công khai và thường dùng làm benchmark cho các mô hình NLP phân loại văn bản.

## Số lớp

Tập dữ liệu có 4 lớp (4 category):

- 0: World (Tin tức thế giới)
- 1: Sports (Thể thao)
- 2: Business (Kinh doanh)
- 3: Sci/Tech (Khoa học & Công nghệ)

## Kích thước

Phiên bản thường dùng (được cung cấp qua Hugging Face `datasets`):

- Train: 120,000 ví dụ
- Test: 7,600 ví dụ

Mỗi ví dụ thường bao gồm hai trường chính: `title` và `description` (tùy nguồn). Khi sử dụng `datasets.load_dataset('ag_news')`, trường `text` là sự kết hợp hoặc phần nội dung mà dataset cung cấp.

## Định dạng

Mỗi bản ghi (record) thường có cấu trúc JSON hay dict với các trường:

- `text` (chuỗi): nội dung văn bản (tiêu đề + mô tả)
- `label` (số nguyên 0-3): nhãn lớp tương ứng

## Tiền xử lý khuyến nghị

- Làm sạch cơ bản: loại bỏ khoảng trắng dư thừa, chuyển về chữ thường nếu dùng embedding không phân biệt hoa-thường.
- Tokenization: Có thể dùng tokenizer đơn giản (split theo khoảng trắng) cho thử nghiệm, nhưng để đạt hiệu năng tốt hơn nên dùng tokenizer học sâu (BPE / WordPiece) hoặc `transformers` tokenizer.
- Truncation / Padding: chuẩn hóa độ dài chuỗi đầu vào (ví dụ `seq_len=128`), pad bằng token `<pad>` và dùng token `<unk>` cho từ lạ.
- Xây dựng vocabulary: nếu dùng embedding học từ đầu, xây vocab dựa trên tập train (giữ top-k từ theo tần suất).

## Các lưu ý

- Dữ liệu ở cấp văn bản tiếng Anh — nếu muốn mở rộng đến các ngôn ngữ khác cần chuyển đổi hoặc dùng datasets tương ứng.
- Dataset đã được tiền xử lý cơ bản; tuy nhiên khi nghiên cứu cần kiểm tra lại phân phối lớp và ví dụ ngoại lệ.
- Lưu ý về bản quyền/nguồn: AG News là tập dữ liệu tổng hợp từ nguồn công khai; khi công bố kết quả, trích dẫn nguồn dataset theo quy định của nơi cung cấp (Hugging Face / tác giả gốc).

## Gợi ý thí nghiệm

- So sánh Pre-Norm vs Post-Norm trên các kiến trúc với độ sâu khác nhau (`n_layers`) để quan sát ổn định gradient.
- Thử các học suất (learning rate) và chiến lược giảm LR (scheduler).
- Sử dụng các kỹ thuật tăng cường dữ liệu văn bản (data augmentation) nếu muốn.
