# Backend Hệ Thống Giám Sát Nông Nghiệp Thông Minh 🌿

Chào mừng bạn đến với backend của hệ thống giám sát nông nghiệp thông minh! Dự án này sử dụng FastAPI để tạo ra một API mạnh mẽ, có khả năng:

* Nhận và lưu trữ dữ liệu từ các cảm biến (nhiệt độ, độ ẩm, độ ẩm đất) do ESP32 gửi lên.
* Nhận hình ảnh cây trồng từ ESP32-CAM.
* Sử dụng model **YOLO** để tự động **phát hiện bệnh** trên lá cà chua từ ảnh nhận được.
* Tích hợp **Google Gemini** để:
    * Đưa ra **lời khuyên tưới tiêu/bón phân** dựa trên dữ liệu cảm biến và dự báo thời tiết (từ OpenWeatherMap).
    * Cung cấp **thông tin chi tiết về bệnh** (triệu chứng, cách điều trị) khi được phát hiện.
* Gửi **thông báo và cảnh báo** (kèm hình ảnh nếu có bệnh) đến **Telegram**.
* Cung cấp API cho phép frontend (ví dụ: dashboard Next.js) **hiển thị dữ liệu**, **điều khiển máy bơm**, và **yêu cầu ESP32-CAM chụp ảnh**.

---

## ✨ Tính Năng Nổi Bật

* **Giám sát thời gian thực:** Nhận dữ liệu cảm biến liên tục.
* **Phát hiện bệnh tự động:** Sử dụng AI (YOLO) để phân tích ảnh cây trồng.
* **Tư vấn thông minh:** Gemini đưa ra gợi ý chăm sóc cây dựa trên dữ liệu và thời tiết.
* **Điều khiển từ xa:** Bật/tắt máy bơm và cài đặt ngưỡng tưới tự động qua API.
* **Cảnh báo tức thì:** Nhận thông báo qua Telegram khi có vấn đề hoặc cập nhật quan trọng.
* **Lưu trữ lịch sử:** Dữ liệu cảm biến và ảnh được lưu lại (SQLite và thư mục).

---

## 🛠️ Công Nghệ Sử Dụng

* **Python 3.10+**
* **FastAPI:** Framework web hiệu năng cao.
* **Uvicorn:** ASGI server để chạy FastAPI.
* **SQLite:** Database gọn nhẹ để lưu dữ liệu cảm biến.
* **Ultralytics (YOLO):** Thư viện nhận diện đối tượng/phân loại ảnh.
* **Google Gemini API:** Mô hình ngôn ngữ lớn cho phân tích và tư vấn.
* **OpenWeatherMap API:** Cung cấp dữ liệu dự báo thời tiết.
* **Telegram Bot API:** Gửi thông báo.
* **HTTPX:** HTTP client bất đồng bộ (cho hiệu năng tốt hơn `requests`).
* **python-dotenv:** Quản lý biến môi trường từ file `.env`.

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy (Local)

### 1. Chuẩn Bị

* Đã cài đặt **Python 3.10** trở lên.
* Đã cài đặt **pip** (thường đi kèm Python).
* Đã cài đặt **Git**.

### 2. Cài Đặt

1.  **Tải code về:** Mở terminal và chạy lệnh:
    ```bash
    git clone [https://github.com/hi3rdt/agricultural-backend.git](https://github.com/hi3rdt/agricultural-backend.git)
    cd agricultural-backend
    ```

2.  **Tạo và kích hoạt môi trường ảo (`venv`):** Việc này giúp cô lập các thư viện của dự án.
    * *Windows:*
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * *macOS/Linux:*
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    Bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh.

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Thiết Lập Biến Môi Trường (API Keys):**
    * Tạo một file mới tên là **`.env`** trong thư mục gốc (`agricultural-backend`).
    * Mở file `.env` và thêm vào các API keys của bạn. **Thay thế các giá trị `YOUR_...` bằng key thật.**
        ```dotenv
        # File .env - Lưu các thông tin bí mật
        TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
        TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
        OPENWEATHER_API_KEY="YOUR_OPENWEATHER_API_KEY"
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

        # Tên file model YOLO (đặt trong thư mục gốc)
        YOLO_MODEL_PATH="yolov12n.pt" # Sửa lại tên file nếu cần

        # URL công khai khi deploy (ví dụ Render/ngrok) - dùng cho webhook Telegram
        WEBHOOK_BASE_URL="[https://your-deployed-url.onrender.com](https://your-deployed-url.onrender.com)"
        ```
    * **QUAN TRỌNG:** Thêm `.env` vào file `.gitignore` để tránh đưa key lên Git: Mở file `.gitignore` (hoặc tạo mới) và thêm dòng `.env`.

5.  **Đặt File Model YOLO:**
    * Đặt file model YOLO đã huấn luyện (ví dụ: `yolov12n.pt`) vào thư mục gốc (`agricultural-backend`), cùng cấp với file `main.py`. Tên file phải khớp với `YOLO_MODEL_PATH` trong file `.env`.

### 3. Chạy Server

Trong terminal (vẫn đang kích hoạt `venv`), chạy lệnh:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
