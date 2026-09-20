# VN-AV Forensics

Hai nhánh cho video một người nói, có audio và thấy rõ miệng:

- **timing:** ước lượng độ lệch có dấu, phát hiện lệch toàn clip/cục bộ.
- **lip_audio_mismatch:** bất nhất môi–âm thanh còn lại sau khi xét căn chỉnh thời gian hợp lý.

Một backbone FATE dùng chung, một model hai head và một checkpoint. Demo không sửa video: chỉ ghép đặc trưng ở các thời điểm tương ứng. Không phân loại kỹ thuật giả mạo, âm vị cụ thể hay danh tính giọng–mặt.

## Ba folder

```text
vn-av-forensics-data        YouTube → tải → cắt/lọc → duyệt clip sạch
vn-av-forensics-generation  chia nhóm → clean + 5 kỹ thuật → duyệt → lưu MP4/nhãn
vn-av-forensics-training    cache FATE → warmup timing → học chung → đánh giá → demo
```

[Tài liệu kiến trúc và các công trình tham khảo](AV_INCONSISTENCY_MODEL_REVIEW_2026-09-14.md).

## 1. Tạo dữ liệu sạch

Từ gốc repo, Anaconda Prompt; cài Git và Node.js vào PATH:

```powershell
conda create -n vn-av-data python=3.11 -y
conda activate vn-av-data
cd vn-av-forensics-data
python -m pip install -e .
python -m vn_av_data setup
```

Tạo `data/sources/dataset_v002/videos.csv` theo [CSV mẫu](vn-av-forensics-data/configs/videos.example.csv), điền URL thật và speaker_id nhất quán. Phiên bản/đợt tải đặt trong `steps/settings.py`.

```powershell
python steps/01_collect.py
python steps/02_download.py
python steps/03_cut.py
python steps/04_merge_manifest.py
python steps/05_review.py
```

Mở http://127.0.0.1:8001, duyệt keep/reject/uncertain và xác nhận tiếng–hình khớp. Dừng bằng Ctrl+C rồi:

```powershell
python steps/06_export.py
python -m vn_av_data validate --dataset exports/dataset_v002
```

Tải theo batch có snapshot nguồn và trạng thái để resume; chạy lại lệnh download bỏ qua file đã xác minh. Có thể thêm `--limit 2`, `--dry-run` hoặc `--cookies-from-browser edge --force-ipv4`. Cut chặn batch chưa tải xong. Giữ nhiều nhóm nguồn/người độc lập; cùng người phải cùng speaker_id. Tối thiểu ba nhóm để chia train/validation/test. Không còn yêu cầu donor khác người trong từng split.

## 2. Tạo dataset bất nhất

| Kỹ thuật | Nhãn timing | Nhãn mismatch |
|---|---|---|
| clean (đối chứng) | 0 | 0 |
| global_lag | Offset đã biết | 0 ở cửa sổ còn đủ nội dung nguồn |
| local_lag | Offset từng đoạn | 0 tại cửa sổ không vắt qua biên |
| sequence_swap | Chưa biết trong đoạn thay | Duyệt bất nhất khi thay câu |
| content_splice | Chưa biết trong đoạn thay | Duyệt bất nhất khi ghép đoạn lời |
| motion_freeze | Chưa biết trong đoạn đóng băng | Duyệt có lời nói nhưng môi đứng |

Bỏ source_swap. Generator tạo đủ ±0,2/0,4/0,6/0,8 giây; sạch là lớp 0. Nhãn mismatch: 1 có bất nhất, 0 đã xác nhận tương thích, -1 chưa biết/không tính loss. Tên generator không tự là nhãn dương. Đứng hình toàn khung là mẫu dễ có dấu hiệu phụ, phải báo kết quả riêng từng kỹ thuật.

Chia nhóm nguồn/người trước tạo biến thể; donor không vượt split. Sạch và bất nhất cùng chính sách encode. File MP4, hash, nguồn, split và nhãn được lưu để train nhiều lần.

**Kaggle:** upload dataset sạch; mở [generate_relations.ipynb](vn-av-forensics-generation/notebooks/generate_relations.ipynb), sửa YOUR_CLEAN_DATASET, bật Internet, chạy CPU. Tải `relations_two_head_v1.zip` rồi giải nén vào `vn-av-forensics-generation/outputs/`.

**Hoặc tạo local**, từ folder generation:

```powershell
conda activate vn-av-data
python -m pip install -e .
python -m vn_av_generation plan --dataset ../vn-av-forensics-data/exports/dataset_v002 --output plans/plan_two_head_v1.json
python -m vn_av_generation render --dataset ../vn-av-forensics-data/exports/dataset_v002 --plan plans/plan_two_head_v1.json --output outputs/relations_two_head_v1
```

Duyệt chung nhãn môi–âm thanh:

```powershell
python -m vn_av_generation review --dataset outputs/relations_two_head_v1
```

Mở http://127.0.0.1:8002. Positive chỉ khi quan sát bất nhất không giải thích được bằng dịch thời gian hợp lý; uncertain nếu chưa rõ. Sửa khoảng đúng theo video. Dừng server rồi:

```powershell
python -m vn_av_generation finalize --dataset outputs/relations_two_head_v1 --review outputs/relations_two_head_v1/review.csv --output outputs/relations_two_head_v1/manifest-reviewed.jsonl
python -m vn_av_generation inspect --dataset outputs/relations_two_head_v1 --manifest manifest-reviewed.jsonl
```

Giữ toàn bộ folder: clips/, manifest.jsonl, dataset_info.json, generation.json, review.csv, manifest-reviewed.jsonl và manifest-reviewed.info.json. Upload thành Kaggle Dataset. Render lại cùng code/plan/output sẽ xác minh rồi bỏ qua mẫu xong; đổi code/plan cần output mới.

### Dùng lại video đã tạo theo năm head

Không cần render lại nếu đã có MP4. Từ folder generation, thay đường dẫn bằng dataset thực tế:

```powershell
python -m vn_av_generation migrate --dataset outputs/relations_v002 --manifest manifest-reviewed.jsonl --output outputs/relations_v002/manifest-two-heads.jsonl
```

Lệnh tạo manifest/receipt mới, giữ video và nhãn gốc. Positive ở một trong sequence/phoneme_viseme/motion_speech trở thành positive chung. Negative chỉ hợp lệ khi đủ ba loại được xác nhận âm tính; phần còn lại chưa biết. Source_swap bị loại khỏi manifest mới, không xóa video. Mẫu lag thuần túy làm negative mismatch ở vùng hợp lệ.

Trong notebook train, đặt DATASET tới folder cũ và MANIFEST thành `manifest-two-heads.jsonl`. Audit sẽ báo thiếu nhãn nếu cần duyệt/bổ sung dữ liệu. Data cũ chỉ có ±0,2/±0,8 vẫn dùng được nhưng chưa phủ đủ lưới; nên tạo phiên bản mới đủ offset. Manifest control ảo cũ chưa có MP4 phải đi qua generator mới.

## 3. Train hai head trên Kaggle

Hai notebook clone repo GitHub: cần đưa code mới lên repo trước, hoặc upload code và sửa đường dẫn. Thay đổi local không tự lên GitHub.

Mở [train_fate_relations.ipynb](vn-av-forensics-training/notebooks/train_fate_relations.ipynb), thêm dataset đã duyệt, sửa DATASET/MANIFEST, bật GPU và Internet:

```text
validate → import → audit → setup → doctor → prepare → train → evaluate → checkpoint-info
```

Import giữ nguyên split, kiểm tra hash và nhãn. Prepare chỉ đọc video đã render, không tạo bất nhất lần nữa. Setup tải code/weights FATE và YuNet. FATE đóng băng; cache đặc trưng dùng lại giữa các epoch.

Một lệnh train chạy 23 epoch tối đa: **3 epoch warmup timing + 20 epoch học chung hai head**. Warmup chưa xuất best.pt dùng demo. Hai giai đoạn dùng cùng model/optimizer; validation/test luôn dùng lag dự đoán, không lấy lag đáp án để căn chỉnh hộ. Resume bằng `python -m vn_av_training train --resume` khi còn last.pt và cấu hình/data không đổi.

Train yêu cầu hai lớp dùng được cho cả hai head trong train và validation. Nhãn thiếu không là negative. Threshold chọn trên validation với giới hạn FAR/precision/recall/AUROC; timing kiểm tra thêm coverage và MAE. Khi lag chưa chắc, mismatch chỉ được đưa ra kết luận nếu nhóm validation tương ứng đã đạt kiểm tra riêng. Nếu không, trả null; điểm thô vẫn xem được ở chế độ chẩn đoán.

Kết quả: `runs/fate-two-head-v1/`, cache: `cache/fate-two-head-v1/`. Xem history.json (phase), validation-report.json và evaluation-test/metrics.json, gồm kết quả theo kỹ thuật. Kiểm thử phần mềm không chứng minh độ chính xác trên video thật; cần đánh giá nguồn/người mới và tiếng Việt riêng.

**Weight năm head cũ không tương thích.** Cần train checkpoint định dạng fate-two-heads-v1; không đổi tên weight cũ để nạp. Cache trích đặc trưng mới nằm ở thư mục riêng do đường xử lý đã đổi; không ghi đè cache/run cũ. Tải ZIP artifact cuối notebook; giữ thêm cache nếu muốn train lại không trích đặc trưng.

## 4. Demo: backend Anaconda + frontend terminal

### Terminal 1 — Anaconda Prompt

```powershell
conda create -n vn-av-training python=3.11 -y
conda activate vn-av-training
cd D:\COMP\RESEARCH\Deepfake_VN\vn-av-forensics\vn-av-forensics-training
python -m pip install -e . -r environments/fate.txt
python -m vn_av_training setup --config configs/relations-local.yaml
```

Giải nén artifact vào folder training để có `runs/fate-two-head-v1/best.pt`; giữ cấu hình encoder như khi tạo cache trên Kaggle. Nếu đã có môi trường tương thích thì activate và cài cập nhật package.

```powershell
python -m vn_av_training checkpoint-info --config configs/relations-local.yaml
python -m vn_av_training doctor --config configs/relations-local.yaml --stage inference --load
python -m vn_av_training serve --config configs/relations-local.yaml
```

Backend ở http://127.0.0.1:8000, local mặc định CPU.

### Terminal 2 — frontend

```powershell
cd D:\COMP\RESEARCH\Deepfake_VN\vn-av-forensics\vn-av-forensics-training\demo
npm install
npm run dev
```

Mở **http://127.0.0.1:5173**. Nếu PowerShell chặn npm.ps1, dùng npm.cmd install và npm.cmd run dev.

UI hiển thị tiến trình thực tế → ảnh/RMS/shape đặc trưng → chất lượng hai head → lag và cặp thời điểm được ghép → timeline/khoảng → tải video bằng chứng và JSON/CSV. Mặc định chỉ xem vùng đủ điều kiện; bật checkbox để xem điểm chẩn đoán. CSV tách cột diagnostic_ khỏi điểm đủ điều kiện. Video gốc không bị chỉnh; RMS/ảnh kiểm tra không phải giải thích âm vị.
