# Hướng dẫn chạy VN-AV Forensics

`vn-av-forensics-data`: tải, cắt, duyệt và xuất dataset; không cần GPU. `vn-av-forensics-training`: train, đánh giá, API và demo; Kaggle dùng GPU, máy local có thể chạy inference bằng CPU.

Quy trình gồm ba phần:

1. Tải video YouTube, cắt clip và duyệt dữ liệu trên máy cá nhân.
2. Upload dataset lên Kaggle, dùng GPU để trích đặc trưng và train.
3. Tải checkpoint về máy, chạy API và frontend để demo.

```text
YouTube -> vn-av-forensics-data -> dataset_v001
        -> Kaggle / vn-av-forensics-training -> best.pt
        -> API FastAPI + frontend React -> http://127.0.0.1:8000
```

## 1. Tải và chuẩn bị dữ liệu trên máy local

### 1.1. Cài pipeline dữ liệu

```powershell
conda create -n vn-av-data python=3.11 -y
conda activate vn-av-data
cd .\vn-av-forensics-data
python -m pip install -e .
python -m vn_av_data setup
```

Lệnh `setup` tải Silero VAD và YuNet vào `checkpoints/media/`. FFmpeg được cung cấp qua `imageio-ffmpeg`.

### 1.2. Khai báo video nguồn

Tạo file `vn-av-forensics-data/data/sources.csv` với mã hóa UTF-8:

```csv
url,speaker_id
https://www.youtube.com/watch?v=VIDEO_ID_1,speaker_01
https://www.youtube.com/watch?v=VIDEO_ID_2,speaker_02
https://www.youtube.com/watch?v=VIDEO_ID_3,speaker_03
```

Quy ước dữ liệu:

- Không nhập URL playlist.
- Các video của cùng một người phải dùng cùng `speaker_id`.
- Nên chọn video một người nói, thấy rõ mặt và miệng, ít chuyển cảnh, không lồng tiếng.
- Dataset train cần ít nhất ba nhóm nguồn/người độc lập để tạo đủ train, validation và test.

### 1.3. Tải video

```powershell
python -m vn_av_data download --sources data/sources.csv --output data/raw
```

Kết quả chính là video trong `data/raw/` và manifest `data/raw/sources.jsonl`. Nếu bị gián đoạn, chạy lại cùng lệnh; lỗi tải được ghi trong `data/raw/download-errors.json`.

Nếu đã có video trên máy, chép video vào `data/raw/` rồi dùng lệnh sau thay cho `download`:

```powershell
python -m vn_av_data index --root data/raw --output data/raw/sources.jsonl
```

Không chạy đồng thời cả `download` và `index` cho cùng một đợt dữ liệu.

### 1.4. Cắt video thành clip ứng viên

```powershell
python -m vn_av_data cut --manifest data/raw/sources.jsonl --output data/candidates/v001
```

Mặc định pipeline dùng VAD, kiểm tra chuyển cảnh và phát hiện mặt để tạo clip dài khoảng 3-8 giây. Kết quả nằm trong `data/candidates/v001/`; xem `summary.json`, `errors.json` và `rejected.json` để kiểm tra đợt xử lý.

### 1.5. Duyệt clip thủ công

```powershell
python -m vn_av_data review --review data/candidates/v001/review.csv --root data/candidates/v001
```

Mở [http://127.0.0.1:8001](http://127.0.0.1:8001). Với mỗi clip:

- Chọn `keep` khi chỉ có người cần thu, thấy rõ miệng, âm thanh đúng người và hình-tiếng khớp tự nhiên.
- Chọn `reject` nếu clip lỗi, nhiều người, che miệng, lồng tiếng hoặc lệch tiếng-hình.
- Chọn `uncertain` nếu chưa chắc chắn.

Quyết định được lưu vào `review.csv`. Nhấn `Ctrl+C` tại PowerShell để dừng server.

### 1.6. Xuất và kiểm tra dataset

```powershell
python -m vn_av_data export --review data/candidates/v001/review.csv --root data/candidates/v001 --output exports/dataset_v001 --dataset-id dataset_v001

python -m vn_av_data validate --dataset exports/dataset_v001
```

Chỉ các clip được duyệt `keep` mới được xuất. Cấu trúc đầu ra:

```text
vn-av-forensics-data/exports/dataset_v001/
|-- clips/
|-- manifest.jsonl
|-- dataset_info.json
`-- quality_report.json
```

Không sửa trực tiếp một dataset đã xuất. Khi thay nguồn hoặc nhãn, xuất một phiên bản mới như `dataset_v002`.

## 2. Train FATE Relation Detector trên Kaggle

Toàn bộ code dùng trên Kaggle nằm trong notebook [train_fate_relations.ipynb](vn-av-forensics-training/notebooks/train_fate_relations.ipynb). Phần này chỉ hướng dẫn thứ tự chuẩn bị và chạy notebook.

### 2.1. Chuẩn bị repo và dataset

Push toàn bộ repo lên [linhxm/vn-av-forensics](https://github.com/linhxm/vn-av-forensics). Repo chứa code của pipeline chuẩn bị dữ liệu và pipeline training; dữ liệu sinh ra, checkpoint, cache và kết quả train không được đẩy lên GitHub vì đã nằm trong `.gitignore`.

Upload thư mục `dataset_v001/` đã xuất thành một Kaggle Dataset riêng. Khi tạo Kaggle Notebook:

1. Thêm Kaggle Dataset chứa `dataset_v001/` vào mục **Input**.
2. Chọn GPU trong **Accelerator**.
3. Bật **Internet** để notebook clone repo và tải FATE cùng pretrained weights.
4. Import notebook `train_fate_relations.ipynb` từ repo GitHub.

Các bước dùng GitHub chỉ thực hiện được sau khi repo đã được push. Nếu repo để private, notebook cần quyền truy cập GitHub; repo public có thể clone trực tiếp.

### 2.2. Chọn đúng đường dẫn dataset

Kaggle chuyển tên dataset thành slug viết thường. Xem đường dẫn thật trong bảng **Input**, sau đó thay `YOUR_DATASET` trong notebook bằng slug đó. Đường dẫn phải trỏ tới thư mục `dataset_v001/` và phép kiểm tra trong notebook phải thành công trước khi train.

URL repo và đường dẫn thư mục training đã được khai báo sẵn trong notebook, không cần nhập lại ở README.

### 2.3. Chạy notebook

Chạy các cell theo thứ tự từ trên xuống. Notebook lần lượt thực hiện:

1. Clone repo vào `/kaggle/working` và cài môi trường FATE.
2. Trỏ cấu hình tới Kaggle Dataset, validate dữ liệu, chia tập và tạo control.
3. Tải đúng phiên bản FATE, pretrained weights và kiểm tra khả năng nạp backbone.
4. Trích đặc trưng FATE vào cache.
5. Train các relation head và lưu checkpoint tốt nhất.
6. Đánh giá trên test split và phân tích thử một clip thuộc test split.
7. Đóng gói checkpoint, metrics, manifest và cấu hình thành `vn-av-forensics-artifacts.zip`.

Nếu notebook yêu cầu restart session sau khi cài thư viện, restart rồi chạy tiếp lại từ cell cấu hình dataset. Khi phiên train bị ngắt, dùng lựa chọn resume đã ghi ngay trong cell training; chỉ resume khi `last.pt` còn tồn tại.

### 2.4. Lưu kết quả

Sau khi cell cuối hoàn tất, tải `vn-av-forensics-artifacts.zip` trong mục **Output** của Kaggle. File quan trọng nhất để chạy inference là `runs/fate-v001/best.pt`.

Gói artifact gọn không chứa feature cache. Nếu muốn tiếp tục ở phiên Kaggle khác mà không trích đặc trưng lại, cần lưu thêm `cache/fate-v001/`, `runs/fate-v001/`, `datasets/manifests/` và `configs/` dưới dạng Kaggle Dataset output.
## 3. Chạy demo frontend trên máy local

### 3.1. Cài pipeline inference

Trở về thư mục gốc của repo và tạo môi trường riêng cho training/demo:

```powershell
conda create -n vn-av-training python=3.11 -y
conda activate vn-av-training
cd .\vn-av-forensics-training
python -m pip install -e . -r environments/fate.txt
```

Tải FATE, pretrained weights và YuNet về đúng vị trí local:

```powershell
python -m vn_av_training setup --config configs/relations-local.yaml
```

Giải nén artifact Kaggle và đặt checkpoint tại:

```text
vn-av-forensics-training/runs/fate-v001/best.pt
```

### 3.2. Build frontend React

```powershell
cd .\demo
npm.cmd ci
npm.cmd run build
cd ..
```

Frontend sau khi build nằm ở `demo/dist/`. Nếu chưa build, backend vẫn có một trang HTML tối giản, nhưng bản React cung cấp đầy đủ giao diện timeline và tải kết quả.

### 3.3. Kiểm tra model và khởi động server

```powershell
python -m vn_av_training doctor `
  --config configs/relations-local.yaml `
  --stage inference `
  --load

python -m vn_av_training serve `
  --config configs/relations-local.yaml
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000), chọn một video có một người nói và nhấn **Phân tích video**. API nhận các định dạng `.mp4`, `.webm`, `.mov`, `.mkv`, `.avi`, `.mpg`; MP4 phù hợp nhất để phát lại trong trình duyệt. Cấu hình mặc định giới hạn video phân tích ở 60 giây.

Local dùng CPU nên lần nạp model và phân tích có thể chậm. Giữ cửa sổ PowerShell đang chạy server; dùng `Ctrl+C` để dừng.

### 3.4. Chạy frontend ở chế độ phát triển

Khi cần sửa giao diện, chạy backend ở một PowerShell và chạy Vite ở PowerShell khác:

```powershell
cd D:\COMP\RESEARCH\Deepfake_VN\vn-av-forensics\vn-av-forensics-training\demo
npm.cmd run dev
```

Mở [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite tự proxy các request `/api` sang backend tại cổng `8000`.
