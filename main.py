import uvicorn
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import logging
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
import httpx
import google.generativeai as genai
import json
import os
import time
from fastapi.staticfiles import StaticFiles
from typing import List
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fastapi.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hệ Thống Giám Sát Nông Nghiệp Thông Minh")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "data.db"
UPLOAD_DIRECTORY = "uploaded_images"
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov12n.pt") # Provide a default

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL")

if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, OPENWEATHER_API_KEY, GEMINI_API_KEY]):
    logger.error("!!! Thiếu các API keys cần thiết trong file .env hoặc biến môi trường! Vui lòng kiểm tra lại.")
    # Consider raising an exception here if keys are critical for startup
    # raise ValueError("Thiếu API keys cần thiết.")

db_lock = Lock()
capture_state = {"capture_requested": False}
gemini_model = None
yolo_model = None

try:
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"Đã tải model YOLO thành công từ: {YOLO_MODEL_PATH}")
    else:
        logger.warning(f"Không tìm thấy model YOLO tại: {YOLO_MODEL_PATH}. Chức năng nhận diện bệnh sẽ bị tắt.")
except Exception as e:
    logger.error(f"Lỗi khi tải model YOLO: {e}", exc_info=True)

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
        logger.info("Đã cấu hình Gemini.")
    else:
        logger.warning("GEMINI_API_KEY bị thiếu. Chức năng Gemini sẽ bị tắt.")
except Exception as e:
    logger.error(f"Lỗi khi cấu hình Gemini: {e}")

def init_db():
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS sensor_data
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, temperature REAL,
                          humidity REAL, soil REAL, pump_status INTEGER, mode TEXT,
                          low_threshold INTEGER, high_threshold INTEGER)''')
            conn.commit()
            conn.close()
            logger.info("Đã khởi tạo database: %s", DB_FILE)
    except Exception as e:
        logger.error("Lỗi khi tạo database: %s", e)
        raise

def init_storage():
    try:
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)
            logger.info(f"Đã tạo thư mục lưu ảnh: {UPLOAD_DIRECTORY}")
    except Exception as e:
        logger.error(f"Lỗi khi tạo thư mục ảnh: {e}", exc_info=True)
        raise

init_db()
init_storage()

app.mount("/images", StaticFiles(directory=UPLOAD_DIRECTORY), name="images")

class SensorData(BaseModel):
    temperature: float
    humidity: float
    soil: float
    pump_status: bool

class ControlRequest(BaseModel):
    mode: str
    low_threshold: int
    high_threshold: int
    pump_status: bool

async def get_weather_forecast(lat: float = 10.8231, lon: float = 106.6297):
    if not OPENWEATHER_API_KEY:
        logger.warning("OPENWEATHER_API_KEY thiếu. Không thể lấy dự báo.")
        return []
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
        data = response.json()
        return [{"date": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d"),
                 "temp": item["main"]["temp"], "humidity": item["main"]["humidity"],
                 "rain_prob": item.get("pop", 0) * 100} for item in data["list"][:40]]
    except httpx.RequestError as e:
        logger.error(f"Lỗi khi gọi API OpenWeather: {e}")
        return []
    except Exception as e:
        logger.error(f"Lỗi không xác định khi lấy dự báo thời tiết: {e}")
        return []

async def analyze_irrigation_and_fertilizer(sensor_data: dict, weather_forecast: list):
    if not gemini_model:
        return {"reason": "Chức năng AI Gemini chưa được cấu hình."}
    prompt = f"""
    Dữ liệu cảm biến: Nhiệt độ {sensor_data['temperature']}°C, Độ ẩm không khí {sensor_data['humidity']}%, Độ ẩm đất {sensor_data['soil']}%.
    Dự báo thời tiết 5 ngày: {weather_forecast}.
    Đề xuất: Giờ tưới tối ưu, ngày bón phân. Trả về JSON: {{"optimal_irrigation_time": "giờ", "fertilizer_day": "ngày", "reason": "lý do"}}
    """
    try:
        response = await run_in_threadpool(gemini_model.generate_content, prompt)
        logger.debug(f"Gemini raw response (irrigation): {response.text}")
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi JSON Gemini (irrigation): {e}. Response: '{response.text}'")
        return {"optimal_irrigation_time": "N/A", "fertilizer_day": "N/A", "reason": f"Lỗi parse JSON từ AI."}
    except Exception as e:
        logger.error(f"Lỗi gọi Gemini (irrigation): {e}", exc_info=True)
        return {"optimal_irrigation_time": "N/A", "fertilizer_day": "N/A", "reason": f"Lỗi gọi API AI."}

async def analyze_tomato_disease(disease_name: str):
    if not gemini_model:
        return "Chức năng AI Gemini chưa được cấu hình."
    prompt = f"""
    Phân tích ngắn gọn về bệnh '{disease_name}' trên cây cà chua. Bao gồm:
    1. Mô tả triệu chứng chính (2-3 câu).
    2. Đề xuất phương pháp điều trị (ưu tiên biện pháp sinh học nếu có, sau đó đến hóa học thông dụng).
    Chỉ trả lời phần phân tích, không thêm lời chào.
    Ví dụ: Triệu chứng: ... Điều trị: ...
    """
    try:
        response = await run_in_threadpool(gemini_model.generate_content, prompt)
        logger.info(f"Đã nhận phân tích bệnh '{disease_name}' từ Gemini.")
        analysis_text = response.text.strip().replace("```", "").strip()
        return analysis_text
    except Exception as e:
        logger.error(f"Lỗi khi gọi API Gemini phân tích bệnh '{disease_name}': {e}", exc_info=True)
        return f"Lỗi khi phân tích bệnh '{disease_name}'."

async def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Thiếu TOKEN hoặc CHAT_ID Telegram. Không thể gửi tin nhắn.")
        return
    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Đã gửi tin nhắn Telegram.")
            else:
                logger.error(f"Gửi Telegram thất bại: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Lỗi khi gửi Telegram: {e}", exc_info=True)

async def send_telegram_photo(image_path: str, caption: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Thiếu TOKEN hoặc CHAT_ID Telegram. Không thể gửi ảnh.")
        return
    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as photo_file:
            files = {'photo': (os.path.basename(image_path), photo_file, 'image/jpeg')}
            payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, data=payload, files=files)
            if response.status_code == 200:
                logger.info(f"Gửi ảnh Telegram thành công: {image_path}")
            else:
                logger.error(f"Gửi ảnh Telegram thất bại: {response.status_code} - {response.text}")
    except FileNotFoundError:
         logger.error(f"Không tìm thấy file ảnh để gửi Telegram: {image_path}")
    except Exception as e:
        logger.error(f"Lỗi khi gửi ảnh Telegram: {e}", exc_info=True)

@app.get("/")
def read_root():
    logger.info("Truy cập Root endpoint (/)")
    return {
        "message": "FastAPI Agriculture System Running",
        "endpoints": {
            "POST /sensor": "ESP32 gửi dữ liệu cảm biến",
            "GET /status": "ESP32 lấy trạng thái điều khiển",
            "GET /data": "Dashboard lấy dữ liệu lịch sử",
            "GET /latest": "Dashboard lấy dữ liệu mới nhất",
            "POST /control": "Dashboard cập nhật cài đặt",
            "POST /upload-image-raw": "ESP32-CAM gửi ảnh (chạy YOLO)",
            "GET /api/images": "Dashboard lấy danh sách ảnh",
            "DELETE /api/images/{filename}": "Dashboard xóa ảnh",
            "GET /images/{filename}": "Xem file ảnh tĩnh",
            "POST /api/capture-request": "Dashboard yêu cầu chụp ảnh",
            "GET /api/cam-command": "ESP32-CAM hỏi lệnh", # Corrected path
            "GET /analysis": "Dashboard lấy phân tích tưới tiêu",
            "POST /telegram/webhook": "Nhận lệnh từ Telegram Bot"
        }
    }

@app.post("/sensor")
async def receive_sensor_data(data: SensorData):
    logger.info(f"Nhận dữ liệu từ ESP32: {data.dict()}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT mode, low_threshold, high_threshold FROM sensor_data ORDER BY id DESC LIMIT 1")
            result = c.fetchone()
            mode, low, high = result if result else ("automatic", 30, 70)
            mode = mode or "automatic"
            low = low if low is not None else 30
            high = high if high is not None else 70

            c.execute('''INSERT INTO sensor_data
                         (timestamp, temperature, humidity, soil, pump_status, mode, low_threshold, high_threshold)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (timestamp, data.temperature, data.humidity, data.soil,
                       int(data.pump_status), mode, low, high))
            conn.commit()
            conn.close()
        logger.info("Lưu dữ liệu ESP32 thành công.")
        return {"message": "Dữ liệu đã được lưu"}
    except Exception as e:
        logger.error(f"Lỗi khi lưu dữ liệu: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")

@app.get("/data")
def get_data(limit: int = 24, offset: int = 0):
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            # Use tuple factory for easier mapping later if needed, but returning tuples is fine
            # conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('''SELECT timestamp, temperature, humidity, soil
                         FROM sensor_data
                         WHERE temperature IS NOT NULL
                         ORDER BY id DESC LIMIT ? OFFSET ?''', (limit, offset))
            records = c.fetchall()
            conn.close()
        records.reverse()
        logger.info(f"Truy xuất {len(records)} bản ghi cho biểu đồ")
        return records # Return list of tuples as expected by the frontend
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu biểu đồ: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi server khi đọc dữ liệu biểu đồ.")


@app.get("/latest")
def get_latest_data():
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''SELECT timestamp, temperature, humidity, soil, pump_status, mode
                         FROM sensor_data WHERE temperature IS NOT NULL
                         ORDER BY id DESC LIMIT 1''')
            latest = c.fetchone()
            conn.close()
        if not latest:
            return {"message": "Chưa có dữ liệu"}
        result = {
            "timestamp": latest[0], "temperature": latest[1], "humidity": latest[2],
            "soil": latest[3], "pump_status": bool(latest[4]), "mode": latest[5] or "automatic"
        }
        logger.info(f"Truy xuất dữ liệu mới nhất: {result['timestamp']}")
        return result
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu mới nhất: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi server khi đọc dữ liệu mới nhất.")

@app.get("/status")
def get_control_status():
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''SELECT mode, low_threshold, high_threshold, pump_status
                         FROM sensor_data ORDER BY id DESC LIMIT 1''')
            latest = c.fetchone()
            conn.close()
        if not latest:
            status = {"mode": "automatic", "low_threshold": 30, "high_threshold": 70, "pump_status": False}
        else:
            status = {
                "mode": latest[0] or "automatic",
                "low_threshold": latest[1] if latest[1] is not None else 30,
                "high_threshold": latest[2] if latest[2] is not None else 70,
                "pump_status": bool(latest[3])
            }
        logger.info(f"Truy xuất trạng thái điều khiển: {status}")
        return status
    except Exception as e:
        logger.error(f"Lỗi khi đọc trạng thái điều khiển: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi server khi đọc trạng thái điều khiển.")

@app.post("/control")
async def update_control(request: ControlRequest):
    logger.info(f"Nhận yêu cầu điều khiển: {request.dict()}")
    try:
        if not (0 <= request.low_threshold <= 100 and 0 <= request.high_threshold <= 100):
            raise HTTPException(status_code=400, detail="Ngưỡng không hợp lệ (0-100)")
        if request.low_threshold >= request.high_threshold:
            logger.warning(f"Ngưỡng thấp ({request.low_threshold}) >= Ngưỡng cao ({request.high_threshold})")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''INSERT INTO sensor_data
                         (timestamp, temperature, humidity, soil, pump_status, mode, low_threshold, high_threshold)
                         VALUES (?, NULL, NULL, NULL, ?, ?, ?, ?)''',
                      (timestamp, int(request.pump_status), request.mode,
                       request.low_threshold, request.high_threshold))
            conn.commit()
            conn.close()
        logger.info("Cập nhật điều khiển thành công.")
        msg = (f"🔔 *Cập Nhật Trạng Thái*\n"
               f"- Chế độ: {request.mode.capitalize()}\n"
               f"- Bơm (Manual): {'Bật' if request.pump_status else 'Tắt'}\n"
               f"- Ngưỡng: {request.low_threshold}% - {request.high_threshold}%")
        await send_telegram_message(msg)
        return {"message": "Cập nhật thành công", "config": request.dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật điều khiển: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi server khi cập nhật điều khiển.")

@app.post("/upload-image-raw/")
async def upload_image_raw(request: Request):
    try:
        image_bytes = await request.body()
        if not image_bytes:
            logger.warning("Upload ảnh: không có dữ liệu.")
            raise HTTPException(status_code=400, detail="Không có ảnh nào được gửi.")

        filename = f"plant_{int(time.time())}.jpg"
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Đã lưu ảnh: {file_path} (Size: {len(image_bytes)} bytes)")

        if yolo_model:
            try:
                results = await run_in_threadpool(yolo_model.predict, file_path)
                if results and results[0].boxes:
                    detected_disease = False
                    disease_name = ""
                    confidence = 0.0
                    for box in results[0].boxes:
                        confidence = float(box.conf[0]) # Ensure float
                        class_id = int(box.cls[0])
                        disease_name = yolo_model.names[class_id]
                        logger.info(f"YOLO Detection: Phát hiện '{disease_name}' với độ tin cậy {confidence:.2f}")

                        if "healthy" not in disease_name.lower() and confidence > 0.5: # Lowered threshold
                            detected_disease = True
                            disease_analysis = await analyze_tomato_disease(disease_name)
                            photo_caption = (f"🚨 *CẢNH BÁO BỆNH CÀ CHUA* 🚨\n\n"
                                             f"Phát hiện: *{disease_name}*\n"
                                             f"Độ tin cậy: *{confidence*100:.1f}%*\n"
                                             f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            await send_telegram_photo(file_path, photo_caption)
                            analysis_message = (f"--- Phân Tích & Điều Trị ---\n"
                                                f"{disease_analysis}")
                            if len(analysis_message) > 4096:
                                analysis_message = analysis_message[:4090] + "..."
                            await send_telegram_message(analysis_message)
                            break
                    if not detected_disease:
                        logger.info("Cây khỏe mạnh hoặc độ tin cậy thấp.")
                        if results[0].boxes: # Check again if boxes exist before accessing
                             first_detection = results[0].boxes[0]
                             caption = (f"✅ *KIỂM TRA CÂY* ✅\n\n"
                                        f"Kết quả: *{yolo_model.names[int(first_detection.cls[0])]}* (Conf: {float(first_detection.conf[0])*100:.1f}%)\n"
                                        f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                             await send_telegram_photo(file_path, caption)
                        else:
                             logger.warning("Không có boxes để báo cáo trạng thái khỏe mạnh.")


                else:
                     logger.warning("Model YOLO không phát hiện đối tượng nào trong ảnh.")
                     await send_telegram_message(f"⚠️ *KHÔNG PHÁT HIỆN* ⚠️\n\nModel YOLO không tìm thấy đối tượng nào trong ảnh.")
            except Exception as e:
                logger.error(f"Lỗi khi chạy model YOLO hoặc gọi Gemini: {e}", exc_info=True)
                await send_telegram_photo(file_path, "Ảnh mới từ camera (LỖI PHÂN TÍCH AI)")
        else:
            logger.warning("Bỏ qua nhận diện YOLO (model chưa được tải).")
            await send_telegram_photo(file_path, f"Ảnh mới từ camera (AI tắt) - {datetime.now().strftime('%H:%M:%S')}")
        return {"message": "Upload ảnh thành công!", "filename": filename}
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi upload ảnh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi server khi upload ảnh: {str(e)}")

@app.get("/api/images", response_model=List[dict]) # Specify return type
async def get_image_gallery():
    images_list = []
    try:
        files = [f for f in os.listdir(UPLOAD_DIRECTORY) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_DIRECTORY, f)), reverse=True)
        for f in files:
            try:
                file_path = os.path.join(UPLOAD_DIRECTORY, f)
                stat = os.stat(file_path)
                images_list.append({
                    "id": f, "url": f"/images/{f}",
                    "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "size": round(stat.st_size / 1024, 1)
                })
            except Exception as stat_err: # Catch potential errors reading file stats
                 logger.error(f"Lỗi khi đọc thông tin file {f}: {stat_err}")
                 # Optionally skip this file or add a placeholder
        logger.info(f"Truy xuất thư viện, tìm thấy {len(images_list)} ảnh.")
        return images_list
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách ảnh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Không thể lấy danh sách ảnh.")

@app.delete("/api/images/{filename}")
async def delete_image(filename: str):
    try:
        if ".." in filename or "/" in filename:
            raise HTTPException(status_code=400, detail="Tên file không hợp lệ.")
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Đã xóa ảnh: {filename}")
            return {"message": "Xóa ảnh thành công", "filename": filename}
        else:
            logger.warning(f"Không tìm thấy ảnh để xóa: {filename}")
            raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")
    except Exception as e:
        logger.error(f"Lỗi khi xóa ảnh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi server khi xóa ảnh.")

@app.post("/api/capture-request")
async def request_capture():
    global capture_state
    capture_state["capture_requested"] = True
    logger.info(">>> Nhận được yêu cầu chụp ảnh từ Dashboard <<<")
    return {"message": "Đã gửi yêu cầu chụp"}

@app.get("/api/cam-command") # Corrected endpoint path
async def get_cam_command():
    global capture_state
    if capture_state["capture_requested"]:
        capture_state["capture_requested"] = False
        logger.info(">>> Gửi lệnh 'capture' đến ESP32-CAM <<<")
        return {"command": "capture"}
    else:
        return {"command": "wait"}

@app.get("/analysis")
async def get_analysis():
    logger.info("Nhận yêu cầu phân tích tưới tiêu...")
    try:
        latest_data = get_latest_data() # This is sync
        if not isinstance(latest_data, dict) or "temperature" not in latest_data:
             logger.warning("Không có dữ liệu cảm biến để phân tích.")
             raise HTTPException(status_code=404, detail="Không có dữ liệu cảm biến.")
        sensor_data = {k: latest_data[k] for k in ["temperature", "humidity", "soil"]}

        weather_forecast = await get_weather_forecast()
        if not weather_forecast:
             logger.warning("Không có dữ liệu thời tiết để phân tích.")
             raise HTTPException(status_code=404, detail="Không có dữ liệu thời tiết.")

        analysis = await analyze_irrigation_and_fertilizer(sensor_data, weather_forecast)
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy phân tích: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi server khi phân tích.")

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received webhook data: {data}")
        if "message" in data and "text" in data["message"]:
            command = data["message"]["text"].strip()
            chat_id = str(data["message"]["chat"]["id"]) # Ensure string comparison
            logger.info(f"Processing command: {command}, chat_id: {chat_id}")

            if chat_id != TELEGRAM_CHAT_ID:
                 logger.warning(f"Bỏ qua lệnh từ chat_id lạ: {chat_id}")
                 return {"status": "ignored"}

            if command == "/analyst":
                with db_lock:
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("SELECT * FROM sensor_data WHERE temperature IS NOT NULL ORDER BY id DESC LIMIT 1")
                    row = c.fetchone()
                    conn.close()
                if row:
                    temperature, humidity, soil = row[2], row[3], row[4]
                    forecast = await get_weather_forecast()
                    analysis = await analyze_irrigation_and_fertilizer({"temperature": temperature, "humidity": humidity, "soil": soil}, forecast)
                    message = (f"*Phân tích tưới tiêu (Gemini)*\n"
                               f"- Độ ẩm đất: {soil}%\n- Nhiệt độ: {temperature}°C\n- Độ ẩm KK: {humidity}%\n"
                               f"- Giờ tưới tối ưu: *{analysis.get('optimal_irrigation_time', 'N/A')}*\n"
                               f"- Ngày bón phân: *{analysis.get('fertilizer_day', 'N/A')}*\n"
                               f"- Lý do: {analysis.get('reason', 'N/A')}")
                    await send_telegram_message(message)
                else:
                    await send_telegram_message("Không có dữ liệu cảm biến gần đây để phân tích.")

            elif command == "/status":
                 latest_data = get_latest_data() # Sync call
                 if isinstance(latest_data, dict) and "temperature" in latest_data:
                      msg = (f"*Trạng Thái Hiện Tại*\n"
                             f"- Thời gian: {latest_data['timestamp']}\n"
                             f"- Nhiệt độ: {latest_data['temperature']}°C\n"
                             f"- Độ ẩm KK: {latest_data['humidity']}%\n"
                             f"- Độ ẩm đất: {latest_data['soil']}%\n"
                             f"- Bơm: {'Bật' if latest_data['pump_status'] else 'Tắt'} (Mode: {latest_data.get('mode', 'N/A')})")
                 else:
                      msg = "Chưa có dữ liệu cảm biến."
                 await send_telegram_message(msg)

            # Add other commands like /pump_on, /pump_off, /auto if needed
            # Remember to call await update_control(...) and await send_telegram_message(...)

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Lỗi khi xử lý webhook Telegram: {e}", exc_info=True)
        # Avoid raising HTTP 500 for webhook errors if possible, Telegram might retry
        return {"status": "error processing webhook"}

@app.on_event("startup")
async def on_startup():
    if GEMINI_API_KEY:
        try:
            models = await run_in_threadpool(genai.list_models) # Use threadpool for sync call
            logger.info(f"Available Gemini models: {[m.name for m in models]}")
        except Exception as e:
            logger.error(f"Không thể liệt kê model Gemini: {e}")

    if TELEGRAM_BOT_TOKEN and WEBHOOK_BASE_URL:
        webhook_url_tg = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_BOT_TOKEN}/setWebhook?url={WEBHOOK_BASE_URL}/telegram/webhook"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(webhook_url_tg)
            logger.info(f"Thiết lập Webhook Telegram: {response.text}")
        except Exception as e:
            logger.error(f"Lỗi khi cài đặt webhook Telegram: {e}")
    else:
        logger.warning("Thiếu TOKEN/CHAT_ID Telegram hoặc WEBHOOK_BASE_URL. Webhook sẽ không được cài đặt.")

    if yolo_model: logger.info("Model YOLO đã sẵn sàng.")
    if gemini_model: logger.info("Model Gemini đã sẵn sàng.")

if __name__ == "__main__":
    logger.info("Khởi động FastAPI server trên [http://0.0.0.0:8080](http://0.0.0.0:8080)")
    # Use string 'main:app' for uvicorn.run when using reload=True
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)