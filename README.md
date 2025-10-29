# Backend Hệ Thống Giám Sát Nông Nghiệp Thông Minh

Đây là phần backend (FastAPI) cho hệ thống giám sát nông nghiệp, nhận dữ liệu từ cảm biến ESP32, điều khiển máy bơm, nhận ảnh từ ESP32-CAM, nhận diện bệnh cây bằng YOLO, và cung cấp API cho dashboard Next.js.

---

## ✨ Tính Năng Chính

* **Nhận Dữ Liệu Cảm Biến:** Lưu trữ nhiệt độ, độ ẩm không khí, độ ẩm đất từ ESP32 vào database SQLite.
* **Điều Khiển:** Cung cấp API để dashboard cài đặt chế độ (tự động/thủ công), ngưỡng độ ẩm, và trạng thái bơm.
* **Nhận Diện Ảnh:** Nhận ảnh từ ESP32-CAM, chạy model YOLO (đã huấn luyện) để phát hiện bệnh trên lá cà chua.
* **Phân Tích AI (Gemini):**
    * Đề xuất lịch tưới/bón phân dựa trên dữ liệu cảm biến và dự báo thời tiết (OpenWeatherMap).
    * Phân tích triệu chứng và đề xuất cách điều trị khi phát hiện bệnh.
* **Thông Báo Telegram:** Gửi cảnh báo (phát hiện bệnh, cập nhật trạng thái) đến Telegram Bot.
* **Cung Cấp API:** Cung cấp dữ liệu lịch sử và trạng thái mới nhất cho frontend (Next.js).

---

## 🛠️ Công Nghệ Sử Dụng

* **Ngôn ngữ:** Python 3.x
* **Framework:** FastAPI
* **Server:** Uvicorn
* **Database:** SQLite
* **AI Nhận Diện Ảnh:** YOLO (Ultralytics)
* **AI Phân Tích:** Google Gemini API
* **Thời tiết:** OpenWeatherMap API
* **Thông báo:** Telegram Bot API
* **HTTP Client (Async):** HTTPX

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy Local

### 1. Yêu Cầu

* **Python:** Phiên bản 3.10 trở lên.
* **pip:** Trình quản lý gói của Python.
* **Git:** Để clone repository.

### 2. Cài Đặt

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/hi3rdt/agricultural-backend.git](https://github.com/hi3rdt/agricultural-backend.git)
    cd agricultural-backend
    ```

2.  **Tạo và Kích Hoạt Môi Trường Ảo (`venv`):**
    * Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Cài Đặt Thư Viện:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Cấu Hình Biến Môi Trường:**
    * Tạo một file tên là `.env` trong thư mục gốc (`agricultural-backend`).
    * Thêm các API keys và cấu hình cần thiết vào file `.env`. **Tuyệt đối không commit file này lên Git.**
        ```dotenv
        TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
        TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
        OPENWEATHER_API_KEY="YOUR_OPENWEATHER_API_KEY"
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        # WEBHOOK_BASE_URL="YOUR_NGROK_URL_OR_RENDER_URL" # Chỉ cần nếu dùng webhook Telegram
        YOLO_MODEL_PATH="yolo_model.pt" # Tên file model YOLO
        ```
    * *Lưu ý:* Code hiện tại đang đọc trực tiếp từ biến trong code. Bạn nên sửa lại code để đọc từ file `.env` (dùng thư viện như `python-dotenv`) hoặc từ biến môi trường hệ thống để bảo mật hơn.

5.  **Tải Model YOLO:**
    * Đặt file model YOLO đã được huấn luyện (ví dụ: `yolo_model.pt`) vào thư mục gốc (`agricultural-backend`). Tên file phải khớp với giá trị `YOLO_MODEL_PATH` bạn đặt.

### 3. Chạy Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
