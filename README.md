# Driver Guard v4 – Giám sát dấu hiệu buồn ngủ của tài xế

Ứng dụng desktop (Windows) dùng camera để phát hiện dấu hiệu buồn ngủ / mệt mỏi của tài xế theo thời gian thực, kèm trợ lý an toàn (RAG + Gemini), gợi ý điểm dừng nghỉ và lập kế hoạch hành trình an toàn.

> Lưu ý: Ứng dụng chỉ đưa ra **cảnh báo dấu hiệu quan sát được** và khuyến nghị an toàn, **không chẩn đoán bệnh**. Mọi con số về lộ trình/thời gian chỉ là ước tính.

---

## Tính năng chính

- **Phát hiện buồn ngủ theo thời gian thực**: EAR (nhắm mắt), MAR (ngáp), gật gù, cúi sâu; tự thích ứng với môi trường ánh sáng.
- **Độ tin cậy**: Watchdog/Heartbeat, phát hiện camera bị che, ghi log lỗi (file + console), quản lý vòng đời worker có xác nhận dừng/khởi động lại.
- **Nhắc dừng nghỉ thông minh** (`fatigue_advisor`): code quyết định khi nào nhắc (điểm + ngưỡng + cooldown), Gemini chỉ diễn đạt lại; mất mạng vẫn dùng câu mẫu offline.
- **Bản đồ & lộ trình** hiển thị trực tiếp trong Tkinter, kèm màn hình giám sát thu nhỏ nổi trên bản đồ.
- **Tìm địa điểm** quanh vị trí: trạm xăng, quán ăn, y tế, điểm dừng chân (Geoapify / Google Places / Overpass), có cache offline.
- **Lập kế hoạch hành trình an toàn** (`trip_planner`): chia chặng, thời điểm nghỉ, giờ đến, cảnh báo lái đêm.
- **Theo dõi hành trình thời gian thực** (`trip_session`): tiến độ, điểm dừng kế tiếp, thời gian lái liên tục, lệch tuyến, tự lập lại kế hoạch.
- **AI Agent** (`trip_agent`, Gemini function calling): tự hỏi lại, gọi công cụ, trả lời có kiểm chứng số liệu.
- **Trợ lý RAG** (`rag_gemini`): nạp tài liệu tăng dần, tìm kiếm lai (vector + BM25 + RRF), trích dẫn nguồn `[n]`, trả lời streaming, hoạt động ngoại tuyến từ kho đã nạp.
- **Nhận diện dấu hiệu bất thường trên khuôn mặt bằng YOLO** (`vision_health`, tùy chọn).

---

## Yêu cầu hệ thống

- Windows 10/11
- Python **3.12**
- Webcam
- Kết nối Internet (cho Gemini, bản đồ, tìm địa điểm). Mất mạng vẫn chạy phần phát hiện buồn ngủ cốt lõi.
- Quyền truy cập vị trí của Windows (nếu muốn dùng định vị thật, xem mục [Kiểm tra định vị](#kiểm-tra-định-vị))

---

## Cài đặt & chạy

Mở PowerShell tại thư mục chứa ứng dụng:

```powershell
# 1. Tạo môi trường ảo (chỉ làm lần đầu)
py -3.12 -m venv drowsy

# 2. Kích hoạt môi trường ảo
.\drowsy\Scripts\Activate.ps1

# 3. Cài thư viện (chỉ làm lần đầu)
python -m pip install -r requirements-rag.txt

# 4. Khai báo API key cho phiên làm việc hiện tại
$env:GEOAPIFY_API_KEY = "<GEOAPIFY_API_KEY_CUA_BAN>"
$env:GEMINI_API_KEY   = "<GEMINI_API_KEY_CUA_BAN>"

# 5. Chạy ứng dụng
py -3.12 main.py

# 6. Thoát môi trường ảo khi dùng xong
deactivate
```

> Nếu PowerShell chặn chạy script, chạy một lần:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Cấu hình bằng file `.env` (khuyến nghị)

Thay vì gõ `$env:...` mỗi lần, tạo file `.env` cạnh `main.py`:

```env
GEMINI_API_KEY=your_gemini_key
GEOAPIFY_API_KEY=your_geoapify_key      # nguồn địa điểm mặc định (miễn phí 3000 request/ngày)
GOOGLE_PLACES_API_KEY=                  # tùy chọn (có tính phí)
POI_PROVIDER=geoapify                   # geoapify | google | overpass (để trống = tự chọn theo key)
POI_OVERPASS_FALLBACK=1                 # đặt 0 để tắt Overpass làm phương án cuối
```

**Không commit `.env` lên Git** (đã có `.gitignore`, hãy chắc chắn nó chứa dòng `.env`).

### Lấy API key

| Dịch vụ | Dùng cho | Link |
|---|---|---|
| Gemini | Trợ lý RAG, agent, diễn đạt cảnh báo | https://aistudio.google.com/apikey |
| Geoapify | Tìm địa điểm (POI) | https://www.geoapify.com |
| Google Places (tùy chọn) | Độ phủ địa điểm tốt hơn | https://console.cloud.google.com |

Không có `GEMINI_API_KEY`: agent và phần diễn đạt AI tắt, phần phát hiện buồn ngủ và form lập kế hoạch ở trang Bản đồ vẫn dùng bình thường.

---

## Cấu trúc thư mục

```
Driver-Drowsiness-Guard-v4-main/
├── main.py                  # Điểm vào chính: GUI + vòng lặp phát hiện
├── settings.py              # Đọc biến môi trường và file .env
├── camera_manager.py        # Quản lý quyền sở hữu camera (lease, retry/backoff)
├── worker_lifecycle.py      # Vòng đời worker + Supervisor (dừng/restart có xác nhận)
├── vision_health.py         # Nhận diện dấu hiệu bất thường bằng YOLO (tùy chọn)
├── fatigue_advisor.py       # Luồng mệt mỏi -> gợi ý dừng nghỉ
├── rag_gemini.py            # Trợ lý RAG (embedding, BM25, trích dẫn)
├── trip_planner.py          # Lập kế hoạch hành trình (logic thuần)
├── trip_agent.py            # AI Agent (Gemini function calling)
├── trip_session.py          # Theo dõi hành trình thời gian thực
├── poi_provider.py          # Tìm địa điểm (Geoapify / Google / Overpass)
├── route_map.py             # Bản đồ & lộ trình trong Tkinter
├── geo_utils.py             # Tiện ích hình học cho tuyến đường
├── mini_monitor.py          # Màn hình giám sát thu nhỏ trên bản đồ
├── ui_theme.py              # Bảng màu, font, widget dùng chung (customtkinter)
├── drowsiness_gui_v3.py     # Bản GUI v3 (giữ lại để tham khảo)
├── drowsiness_gui_v3_rag.py # Bản GUI v3 có RAG (giữ lại để tham khảo)
├── face_landmarker.task     # Model MediaPipe face landmarker
├── models/                  # Model bổ sung (vd. health_yolo.pt – tùy chọn)
├── documents/               # Tài liệu để nạp vào RAG
├── chroma_db/               # Kho vector của RAG (tự sinh)
├── requirements-rag.txt     # Danh sách thư viện
├── test_location.py         # Kiểm tra định vị Windows
├── test_worker_lifecycle.py # Test vòng đời worker
├── DrowsinessMonitor.spec   # Cấu hình PyInstaller
├── run_template.bat         # Mẫu file chạy nhanh
└── (tự sinh khi chạy) alerts_log.csv, system_errors.log, poi_cache.json, build/, dist/
```

---

## Dùng trợ lý RAG

1. Đặt tài liệu an toàn lái xe (`.md`, `.docx`, `.pdf`, …) vào thư mục `documents/`.
2. Khi chạy, hệ thống chỉ nạp **file mới / đã đổi / đã xóa** (so sánh SHA-256), không embed lại toàn bộ.
3. Hỏi trong giao diện trợ lý; câu trả lời có trích dẫn `[n]` kèm nguồn (tệp, trang, mục).
4. Mất mạng hoặc Gemini lỗi: vẫn hiện các đoạn liên quan từ kho đã nạp (BM25).

Để đặt lại kho vector, đóng ứng dụng và xóa thư mục `chroma_db/`.

---

## Nhận diện dấu hiệu bất thường trên khuôn mặt (tùy chọn)

Model YOLO COCO có sẵn **không** biết các lớp như "chấn thương", "mắt nhợt nhạt"… Bạn cần tự huấn luyện (hoặc dùng) một model có lớp tương ứng:

1. Đặt model tại `models/health_yolo.pt`.
2. Khai báo tên lớp trong `CLASS_MAP` của `vision_health.py`.
3. Cài `ultralytics` (`python -m pip install ultralytics`).

Nếu thiếu `ultralytics` hoặc thiếu file model, module tự tắt và hệ thống phát hiện buồn ngủ vẫn chạy bình thường.

---

## Kiểm tra định vị

Script `test_location.py` kiểm tra Windows có cấp quyền và trả vị trí hay không:

```powershell
python -m pip install winsdk
python test_location.py
```

Kết quả mong đợi là in ra `Latitude`, `Longitude` và `Accuracy`. Nếu báo chưa cấp quyền, vào **Settings → Privacy & security → Location** và bật cho ứng dụng desktop. Nếu chỉ có vị trí IP (sai số vài km), hệ thống sẽ chuyển sang ước tính theo thời gian và cảnh báo độ tin cậy thấp.

---

## Chạy test

```powershell
python test_worker_lifecycle.py
```

---

## Đóng gói thành file .exe

Dự án có sẵn `DrowsinessMonitor.spec` cho PyInstaller:

```powershell
python -m pip install pyinstaller
pyinstaller DrowsinessMonitor.spec
```

File chạy nằm trong thư mục `dist/`. Nhớ đặt `.env` (hoặc khai báo biến môi trường) cùng nơi chạy file `.exe`, và đảm bảo `face_landmarker.task`, `models/`, `documents/` được đi kèm theo cấu hình trong file `.spec`.

---

## File log & dữ liệu sinh ra

| File | Nội dung |
|---|---|
| `alerts_log.csv` | Lịch sử các cảnh báo buồn ngủ |
| `system_errors.log` | Log lỗi hệ thống |
| `poi_cache.json` | Cache địa điểm (dùng khi mất mạng) |
| `chroma_db/` | Kho vector của RAG |

---

## Xử lý sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Không mở được camera | Đóng ứng dụng khác đang dùng camera; kiểm tra quyền camera của Windows. `camera_manager` sẽ tự retry, vượt giới hạn sẽ báo không phục hồi được. |
| Không có kết quả địa điểm | Kiểm tra `GEOAPIFY_API_KEY`; hệ thống sẽ dùng cache hoặc Overpass nếu có. |
| Trợ lý/agent không phản hồi | Kiểm tra `GEMINI_API_KEY` và kết nối mạng. |
| `Activate.ps1` bị chặn | Chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| Vị trí không chính xác | Bật Location của Windows; GPS/Wi-Fi cho độ chính xác tốt hơn IP. |

---

## Bảo mật

- **Không** ghi API key trực tiếp vào mã nguồn, README hay đẩy lên Git.
- Dùng `.env` (đã nằm trong `.gitignore`) hoặc biến môi trường.
- Nếu lỡ để lộ key, hãy thu hồi và tạo key mới ngay.




