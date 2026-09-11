# Thiết kế bàn phân tích VN-AV Forensics

## Mục tiêu và luồng nghiên cứu

Đưa người nghiên cứu từ một khoảng bất nhất đến bằng chứng có thể kiểm tra: tải video → phân tích → chọn khoảng → xem trong ngữ cảnh → lặp/phát chậm/phóng to → ghi nhận quan sát → xuất dữ liệu và clip.

## Bố cục triển khai

- Thanh đầu: thương hiệu, không gian nghiên cứu, chuyển Việt/Anh, chọn video mới.
- Trước phân tích: khu vực tải/kéo thả video, xem trước, tiêu chí đầu vào và hướng dẫn ba bước. Không hiển thị số liệu giả.
- Trong phân tích: tiến trình theo giai đoạn thực tế, lỗi có thể thử lại, vẫn xem được video đã chọn.
- Sau phân tích: hàng tóm tắt kết quả, số khoảng, độ phủ cửa sổ hợp lệ, độ lệch toàn cục đã bù.
- Cột trái: danh sách khoảng theo thời gian, điểm cao nhất trong các cửa sổ hợp lệ, trạng thái đã xem, lọc chưa xem.
- Trung tâm: một trình phát duy nhất; chọn khoảng để giới hạn phát và lặp, chọn ngữ cảnh 0/0,75/1,5 giây, tốc độ 0,25×/0,5×/1×, tiến/lùi 0,04 giây, phóng 1–4× và chọn tâm phóng bằng con trỏ. Có chế độ toàn video.
- Timeline ngay dưới trình phát: điểm model, đường ngưỡng, vùng nghi vấn, khoảng dữ liệu không hợp lệ, con trỏ đồng bộ và thanh tua dùng được bằng bàn phím. Năng lượng âm thanh hiển thị riêng, chỉ mang nghĩa chất lượng dữ liệu.
- Cột phải: điểm cửa sổ gần thời điểm đang xem (chỉ trong phạm vi lấy mẫu), thông số khoảng, ghi chú, đánh dấu đã xem, tải MP4 và xuất JSON/CSV.
- Màn hình nhỏ: lần lượt danh sách đoạn → trình phát/timeline → ghi chú, không ép ba cột.

## Nguyên tắc bằng chứng

Điểm model không phải xác suất deepfake. Khoảng không hợp lệ/ngoài timeline không được gán điểm gần nhất từ xa. Độ phủ là valid_windows / total_windows trong phần được phân tích, không phải tỷ lệ toàn bộ video. Zoom chỉ phóng pixel nguồn và không tăng chi tiết. Bước 0,04 giây là bước thời gian, không tuyên bố khớp frame của video VFR. Video xem là nguồn gốc; model đã bù độ lệch toàn cục, nên độ lệch nghe/thấy trên nguồn cần diễn giải cùng thông số này. Không suy diễn heatmap hoặc vùng mặt từ dữ liệu API chưa cung cấp.

## Phạm vi bản này

Triển khai bố cục và công cụ kiểm tra trên dữ liệu API thật, cắt MP4 theo khoảng ở backend bằng FFmpeg, ghi chú theo job trong trình duyệt, xuất báo cáo JSON cùng thiết lập xem. Khi chọn file mới phải xóa kết quả cũ khỏi màn hình. Lỗi xuất clip không làm mất kết quả. Clip xuất giữ tốc độ và hình ảnh gốc, bao gồm ngữ cảnh đã chọn; zoom/phát chậm chỉ phục vụ xem.

## Các bước tiếp theo cần thêm dữ liệu backend

1. Cặp video nguồn và face crop có ánh xạ timestamp rõ ràng, bản xem đã bù lag để đối chiếu đúng đầu vào model.
2. Metadata FPS/timebase và chỉ mục frame để bước từng frame chính xác; waveform PCM để đối chiếu âm thanh sâu hơn.
3. Hồ sơ nghiên cứu lưu phía máy chủ: tên mẫu, phiên bản checkpoint/hash, tham số tiền xử lý, nhận xét nhiều người và lịch sử thay đổi.
4. So sánh hai lần chạy, xuất gói bằng chứng gồm clip, metadata, timeline và ghi chú.

## Kiểm tra chấp nhận

Build TypeScript; kiểm tra ranh giới timeline/cửa sổ hợp lệ; clip xuất đúng khoảng và được cache; kiểm tra giao diện rỗng, đang xử lý, lỗi, không đủ dữ liệu, có/không có khoảng; thao tác chọn khoảng, tua ngoài khoảng, loop, zoom, ghi chú và responsive. Ghi rõ những kiểm tra không chạy được do môi trường hoặc thiếu checkpoint/video.

## Kết quả kiểm tra bản triển khai (11/09/2026)

- `npm.cmd run build`: TypeScript và Vite build thành công.
- `npm.cmd test`: 3 kiểm thử qua về cửa sổ ngoài phạm vi, điểm không hợp lệ và giới hạn ngữ cảnh.
- `py -m unittest discover -s tests -v`: 3 kiểm thử qua; chạy FFmpeg thật trên video/âm thanh tổng hợp, kiểm tra thời lượng MP4, giữ track âm thanh, cache và dọn file khi lỗi.
- Chrome headless ở 1440 px và 390 px: kiểm tra chọn file, trạng thái xử lý, phát, lặp/dừng tại biên đoạn, zoom, ghi chú vào localStorage, lọc chưa xem, tua ngoài đoạn, lỗi xuất clip không mất ghi chú, Việt/Anh, thay file xóa kết quả cũ và các trạng thái không có khoảng/không đủ dữ liệu/lỗi. Không có exception JavaScript hoặc tràn ngang trên mobile.
- Ảnh chụp nằm trong `outputs/frontend-review/`; script kiểm tra cục bộ là `outputs/browser-check.mjs`. Kiểm tra trình duyệt dùng video và kết quả có sẵn trong outputs, với API job giả lập để chủ động kiểm tra từng trạng thái; không phải một lần chạy inference mới.
- Chưa kiểm tra HTTP tích hợp trên FastAPI thật trong lượt này: Python mặc định thiếu FastAPI và imageio-ffmpeg. Backend triển khai cần môi trường đã cài dependency từ `pyproject.toml` và khởi động lại để nhận endpoint cắt clip mới. Kiểm thử FFmpeg dùng binary đã có trong `checkpoints/syncnet/bin`.
