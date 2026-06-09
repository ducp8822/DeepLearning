
# Fine-tune GPT-2 cho Toán Tiếng Việt

Project này fine-tune mô hình GPT-2 tiếng Việt cho bài toán giải toán ngắn gọn, với định dạng đầu ra bắt buộc:

`Đáp án là: <số>`

Notebook chính: `group-15-fine-tune-gpt-2-for-math.ipynb`

## 1. Mục tiêu

- Huấn luyện GPT-2 để sinh lời giải ngắn cho bài toán tiếng Việt.
- Tối ưu khả năng trích xuất đáp án số ở cuối câu trả lời.
- So sánh 2 checkpoint (`FINAL` và `BEST_TRAIN_LOSS`) trên tập validation, sau đó tự động chọn model tốt hơn theo điểm số.

## 2. Cấu trúc thư mục

Trong workspace hiện tại:

- `group-15-fine-tune-gpt-2-for-math.ipynb`: Notebook huấn luyện + đánh giá + suy luận test.
- `README.md`: Tài liệu dự án.
- `test_predictions.json`: Ví dụ file dự đoán đầu ra.

## 3. Pipeline trong notebook

Notebook gồm 8 cell code chính:

1. **Thiết lập môi trường & hyperparameters**
- Set seed, cấu hình offline Hugging Face, giới hạn thời gian train.
- Đặt template prompt:
	- `### Bài toán`
	- `### Yêu cầu`
	- `### Lời giải`

2. **Nạp và làm sạch dữ liệu**
- Đọc `train.json`, `valid.json` (hoặc `val.json`).
- Chuẩn hóa response (`normalize_response`).
- Trích đáp án số (`extract_answer_fast`).
- Loại bản ghi lỗi/rỗng/quá dài.
- Khử trùng lặp theo câu hỏi và giữ mẫu chất lượng cao hơn (`quality_score`).

3. **Rút gọn lời giải trước khi train (Response Pruning)**
- Giữ phần cuối của lời giải (mặc định 3 dòng trước dòng đáp án).
- Mục tiêu: chuỗi ngắn hơn, train nhanh hơn, tập trung vào phần kết luận đáp án.

4. **Tạo dataset SFT**
- Tạo cặp prompt-target từ dữ liệu đã làm sạch.
- Build `input_ids`, `labels`, `answer_mask`.
- Áp dụng truncate thông minh để ưu tiên giữ phần đáp án cuối.

5. **Huấn luyện model**
- Dùng lớp `GPT2ForMath` custom loss, tăng trọng số token thuộc dòng đáp án (`ANSWER_WEIGHT = 3.0`).
- Có callback giới hạn thời gian train (`TimeLimitCallback`).
- Có callback lưu checkpoint có train loss tốt nhất (`BestTrainLossCallback`).
- Nếu CUDA OOM thì tự fallback batch size nhỏ hơn.

6. **Đánh giá trên validation**
- Sinh output hoàn toàn bằng model (không fuzzy bypass).
- Chấm điểm theo sai số tương đối giữa số dự đoán và số chuẩn:
	- `10`: sai số <= 1%
	- `5`: sai số <= 10%
	- `1`: sai số <= 50%
	- `0`: còn lại
- So sánh các candidate model và chọn model tốt nhất theo thứ tự ưu tiên:
	- `raw_score` -> `exact` -> `close` -> `partial` -> ưu tiên `FINAL` nếu hòa.

7. **Xuất báo cáo validation**
- `valid_output.json`: output model trên từng mẫu valid.
- `valid_report.json`: tổng hợp điểm, bucket điểm, score theo type, kết quả chọn model.

8. **Suy luận tập test**
- Chạy inference cho cả:
	- `FINAL_MODEL_DIR`
	- `BEST_LOSS_MODEL_DIR`
- Xuất:
	- `test_predictions.json`
	- `test_predictions_best_loss.json`

## 4. Cấu hình chính (theo notebook)

- `MAX_LEN = 256`
- `MAX_TRAIN_SAMPLES = 80000`
- `LR = 1e-3`
- `PER_DEVICE_BS = 16`
- `GRAD_ACCUM = 1`
- `ANSWER_WEIGHT = 3.0`
- `TRAIN_TIME_LIMIT = 165` phút
- `MAX_NEW_TOKENS = 192`
- `GEN_BATCH_SIZE = 24`
- `PRUNE_KEEP_LINES = 3`

## 5. Yêu cầu dữ liệu

Notebook kỳ vọng dữ liệu đặt trong `DATA_DIR` (môi trường Kaggle), gồm:

- `train.json`
- `valid.json` hoặc `val.json`
- `test.json`

Mỗi record dùng các trường chính:

- `id` (khuyến nghị)
- `query_vi`
- `response_vi` (cho train/valid)
- `type` (khuyến nghị để thống kê)

## 6. Cách chạy

1. Mở notebook `group-15-fine-tune-gpt-2-for-math.ipynb`.
2. Đảm bảo model gốc GPT-2 tiếng Việt và dataset đã mount đúng đường dẫn Kaggle như notebook cấu hình.
3. Chạy lần lượt từ cell 1 đến cell 8.
4. Thu các file kết quả trong thư mục làm việc (`/kaggle/working`).

## 7. Đầu ra quan trọng

- Model sau train:
	- `/kaggle/working/gpt2-math-finetuned`
	- `/kaggle/working/gpt2-math-finetuned-best-loss`
- Báo cáo valid:
	- `/kaggle/working/valid_output.json`
	- `/kaggle/working/valid_report.json`
- Dự đoán test:
	- `/kaggle/working/test_predictions.json`
	- `/kaggle/working/test_predictions_best_loss.json`

## 8. Ghi chú

- Notebook thiết kế theo hướng **MODEL-ONLY** (không dùng heuristic bypass khi dự đoán).
- Định dạng đáp án cuối rất quan trọng để chấm điểm ổn định:
	- `Đáp án là: <số>`
- `test_predictions.json` trong repo hiện tại là một kết quả mẫu đã được xuất.

