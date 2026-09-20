# Demo hai nhánh

Một trang đơn giản: đầu vào → tiến trình thật → đặc trưng chung → chất lượng hai head → kết quả và bằng chứng. Chạy backend Anaconda cổng 8000 và Vite bằng npm install / npm run dev cổng 5173.

- Hai nhánh duy nhất: timing và lip_audio_mismatch. Không hiển thị kết luận identity, âm vị cụ thể hay loại generator.
- FATE trích đặc trưng một lần; timing ước lượng lag, mismatch sử dụng cặp gốc/cặp căn chỉnh và confidence. Không sửa file video.
- Hiển thị cửa sổ, bước thời gian, số vùng hợp lệ, shape audio/visual, ảnh kiểm tra và RMS.
- Timeline có hai chế độ tách biệt: vùng đủ điều kiện và điểm chẩn đoán. Khoảng thiếu bằng chứng không được nối lại hay coi là âm tính.
- Hiển thị thời điểm audio/hình được ghép và độ tin cậy lag. Không tìm được lag không tự chứng minh mismatch; nhóm chưa căn chỉnh phải đạt validation riêng.
- Link tải bằng chứng dùng đúng khoảng của chế độ đang xem. JSON/CSV lưu riêng điểm được chấp nhận và điểm thô.
- Không tạo waveform, transcript, phoneme hay heatmap giả. Điểm bất nhất không phải xác suất giả mạo.

Hướng dẫn chạy duy nhất nằm ở README gốc.
