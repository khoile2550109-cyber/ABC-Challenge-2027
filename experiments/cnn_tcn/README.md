# ABC2027 CNN–TCN experiment

Implementation riêng cho baseline thử nghiệm CNN–TCN. Không sửa baseline energy/logistic hoặc evaluator chính thức. Config là điểm bắt đầu chưa tối ưu; chưa có số đo VRAM/thời gian và chưa có kết quả V1/V2.

> **Trạng thái bàn giao:** đã rà soát tĩnh; chưa thực thi, chưa xác nhận chạy thành công. Các lệnh bên dưới chỉ dành cho người chủ trì chạy sau.

## Cấu trúc dữ liệu cần có

DATA_ROOT là thư mục gói phát hành chỉ đọc, ví dụ:

```text
<DATA_ROOT>/
  streams/
    train/stream_00.h5 ... stream_12.h5
    val/stream_00.h5 ... stream_04.h5
  streams_events.csv
  src/abc2027/evaluate.py
```

Mỗi HDF5 cần có cir float32 dạng (3,T,368), t có T timestamp giây và radar_serials gồm ba radar. Mã kiểm tra thứ tự radar theo config, t tăng đơn điệu, shape CIR và nhãn train/validation. Timestamp được đổi về mốc tương đối đầu stream để khớp schema event/submission. Khoảng sự kiện dùng quy ước nửa mở [t_start,t_end): thời gian nằm yên sau khi ổn định trên sàn không được gán fall.

Chỉ đọc streams/train để fit normalizer/train và streams/val để chọn checkpoint/chấm. Không dùng segments/, test, dữ liệu nguồn LIARA hoặc ground truth trong inference. Vùng ignore và hai giây evaluator dùng để nới vùng bỏ qua được mask trong loss; padding không góp loss. Nếu cửa sổ chỉ chứa một phần sự kiện, fall target phản ánh phần giao nhau; start/end target chỉ tạo khi boundary thật nằm trong cửa sổ, không tạo boundary giả ở mép cửa sổ.

## Mô hình và cấu hình

- CNN 1D dùng chung trên range-bin cho mỗi radar-frame, đầu ra 64 chiều/radar.
- Nối theo radar order thành (B,L,192), rồi TCN theo thời gian.
- V1 xuất fall logit (B,L).
- V2 xuất fall/start/end logits, mỗi đầu (B,L); start/end dùng Gaussian soft target với sigma khởi đầu 0,5 giây.
- V1/V2 dùng cùng seed, cửa sổ 400 frame, hop 50 frame, CNN/TCN và cấu hình train. Khác biệt chính là boundary heads/loss.
- HDF5 được đọc theo lát; DataLoader worker tự mở handle riêng. Normalizer chỉ fit từ train, được lưu thành normalizer.npz và đóng gói cùng checkpoint.
- Threshold, merge gap, min duration và boundary search radius đọc từ config. Đây là giá trị khởi đầu, không phải tham số đã tối ưu. merge_gap_s=0.8 và min_duration_s=0.6 được mượn làm điểm khởi đầu từ energy baseline README, chưa hiệu chỉnh riêng cho CNN–TCN.

## Thiết lập môi trường riêng

Các lệnh mẫu cho Linux với Python 3.10 trở lên. Đặt các biến theo máy thuê; OUTPUT_DIR phải nằm ngoài DATA_ROOT.

```bash
export REPO_ROOT=/workspace/ABC-Challenge-2027
export CNN_TCN_ROOT="$REPO_ROOT/experiments/cnn_tcn"
export DATA_ROOT=/mnt/data/abc2027_uwb_train_v1
export EVALUATOR_ROOT="$DATA_ROOT"
export OUTPUT_DIR=/mnt/output/abc2027-cnn-tcn
export ENV_DIR=/mnt/output/venvs/abc2027-cnn-tcn
set -o pipefail
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
```

Chọn lệnh cài PyTorch khớp OS/Python/compute platform của máy theo [PyTorch Get Started](https://pytorch.org/get-started/locally/), rồi cài dependency riêng:

```bash
python -m pip install --upgrade pip
python -m pip install -r "$CNN_TCN_ROOT/requirements.txt"
python -m pip install -r "$CNN_TCN_ROOT/requirements-dev.txt"
export PYTHONPATH="$CNN_TCN_ROOT/src:$EVALUATOR_ROOT/src:$PYTHONPATH"
```

Ghi lại phiên bản môi trường vào output:

```bash
mkdir -p "$OUTPUT_DIR"
python --version | tee "$OUTPUT_DIR/python-version.txt"
python -m pip freeze > "$OUTPUT_DIR/pip-freeze.txt"
```

Gói experiment dùng NumPy, h5py, PyTorch; requirements-dev bổ sung pytest. Không cần cài/sửa dependency trong repo gốc. Ghi lại python --version và python -m pip freeze trong thư mục output để đính kèm báo cáo.

## Kiểm tra dữ liệu và test

```bash
python -m cnn_tcn inspect \
  --config "$CNN_TCN_ROOT/configs/v1.json" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR/data-check"
pytest "$CNN_TCN_ROOT/tests"
```

inspect ghi data_check.json vào output-dir; không ghi vào DATA_ROOT. Test dùng fixture HDF5 tạm dưới pytest temp directory, không dùng bộ dữ liệu phát hành.

## Train V1

```bash
mkdir -p "$OUTPUT_DIR/v1"
python -m cnn_tcn train \
  --variant v1 \
  --config "$CNN_TCN_ROOT/configs/v1.json" \
  --evaluator-root "$EVALUATOR_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR/v1" 2>&1 | tee "$OUTPUT_DIR/v1/train.log"
```

## Train V2

```bash
mkdir -p "$OUTPUT_DIR/v2"
python -m cnn_tcn train \
  --variant v2 \
  --config "$CNN_TCN_ROOT/configs/v2.json" \
  --evaluator-root "$EVALUATOR_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR/v2" 2>&1 | tee "$OUTPUT_DIR/v2/train.log"
```

Mỗi epoch chạy inference offline trên streams/val, ghi metric evaluator chính thức với --ignore-flagged, rồi chọn checkpoint theo ranking: mean F1 tại tIoU 0.1/0.3/0.5; tie-break bằng median absolute start error ở 0.3, sau đó false alarms/giờ ở 0.3. Không dùng test để chọn model. Threshold trong config là operating point ban đầu; nếu thử tuning, chỉ dùng validation và ghi lại mọi lựa chọn.

## Resume

Dùng cùng config, DATA_ROOT và output-dir như lần train trước. Resume kiểm tra config khớp và khôi phục model, optimizer, scaler, epoch, normalizer cùng RNG state.

```bash
python -m cnn_tcn train \
  --variant v1 \
  --config "$CNN_TCN_ROOT/configs/v1.json" \
  --evaluator-root "$EVALUATOR_ROOT" \
  --resume "$OUTPUT_DIR/v1/last.pt" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR/v1" 2>&1 | tee -a "$OUTPUT_DIR/v1/train.log"
```

Với V2, đổi variant/config/output/checkpoint sang thư mục v2.

## Inference validation

Inference chỉ nhận split val, không đọc CSV nhãn và không xuất t_alarm. Nó dùng config và normalizer đóng gói trong checkpoint.

```bash
python -m cnn_tcn infer \
  --checkpoint "$OUTPUT_DIR/v1/best.pt" \
  --split val \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR/v1/inference"
```

File dự đoán là val_predictions.csv với đúng các cột stream_id,t_start,t_end,score; manifest ghi checkpoint, split và xác nhận không ghi t_alarm.

## Chấm bằng evaluator chính thức

Evaluator chính thức được nạp từ EVALUATOR_ROOT/src/abc2027/evaluate.py trong lúc train để chọn checkpoint. Để tạo báo cáo độc lập sau inference, gọi evaluator không sửa đổi với thư mục stream chỉ chứa validation:

```bash
python -m abc2027.evaluate \
  --gt "$DATA_ROOT/streams_events.csv" \
  --pred "$OUTPUT_DIR/v1/inference/val_predictions.csv" \
  --streams "$DATA_ROOT/streams/val" \
  --ignore-flagged \
  --out "$OUTPUT_DIR/v1/inference/official_report.json"
```

--streams "$DATA_ROOT/streams/val" giới hạn stream_id và durations hợp lệ cho validation; evaluator lọc event CSV theo các stream đó. Không truyền streams/ cha vì có thể gồm cả train và val.

## File được tạo dưới OUTPUT_DIR

- data-check/data_check.json: inventory train/validation, shape/timestamp checks và số nhãn.
- v1/ hoặc v2/: config.json, normalizer.npz, last.pt, best.pt, metrics.jsonl.
- v1/validation/ hoặc v2/validation/: dự đoán và report evaluator của epoch gần nhất; chỉ dùng val.
- v1/inference/ hoặc v2/inference/: val_predictions.csv, prediction_manifest.json và report evaluator độc lập.
- train.log, python-version.txt, pip-freeze.txt: log và thông tin môi trường do người chạy thu thập.

Không có code nào trong package tạo file dưới DATA_ROOT. Không lưu checkpoint, cache, prediction hoặc report vào DATA_ROOT. Chỉ công bố số V1/V2 sau khi người chủ trì chạy và lưu kèm config, commit, log và report.

## Rà soát và giới hạn hiện tại

Đã rà soát tĩnh; chưa thực thi, chưa xác nhận chạy thành công. Chưa xác minh tương thích phiên bản trên máy GPU thuê, hiệu năng/VRAM, tốc độ, tính ổn định AMP, điểm validation hoặc kết quả test. Các cấu hình khởi đầu chưa được tune. Baseline energy/logistic và evaluator chính thức được giữ nguyên.


