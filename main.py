from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import logging
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
import requests
import google.generativeai as genai
import json

#  logging 
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fastapi.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

#  Next.js va ESP32 truy cap API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database SQLite
DB_FILE = "data.db"
db_lock = Lock()

#  database 
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS sensor_data
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      temperature REAL,
                      humidity REAL,
                      soil REAL,
                      pump_status INTEGER,
                      mode TEXT,
                      low_threshold INTEGER,
                      high_threshold INTEGER)''')
        conn.commit()
        conn.close()
        logger.info("Đã khởi tạo database: %s", DB_FILE)
    except Exception as e:
        logger.error("Lỗi khi tạo database: %s", e)
        raise

init_db()

#  ESP32
class SensorData(BaseModel):
    temperature: float
    humidity: float
    soil: float
    pump_status: bool

#  dashboard
class ControlRequest(BaseModel):
    mode: str
    low_threshold: int
    high_threshold: int
    pump_status: bool

# API Keys 
TELEGRAM_BOT_TOKEN = "8293702102:AAFPJgSDjLyYtTxamqjAjGjC52FQtyys2kA"
TELEGRAM_CHAT_ID = "-4879272337"  
OPENWEATHER_API_KEY = "02ff7531ae951a7efa49bc9cd0b418d7"
GEMINI_API_KEY = "AIzaSyDBJYHLrAX-W-7weZ3VgseTUeVbJTixwdM"

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-2.0-flash')  

# OpenWeatherMap
async def get_weather_forecast(lat: float = 10.8231, lon: float = 106.6297):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [{"date": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d"),
                 "temp": item["main"]["temp"],
                 "humidity": item["main"]["humidity"],
                 "rain_prob": item.get("pop", 0) * 100} for item in data["list"][:5 * 8]]
    return []

# Phân tích từ Gemini
async def analyze_irrigation_and_fertilizer(sensor_data: dict, weather_forecast: list):
    prompt = f"""
    Dữ liệu cảm biến: Nhiệt độ {sensor_data['temperature']}°C, Độ ẩm không khí {sensor_data['humidity']}%, Độ ẩm đất {sensor_data['soil']}%.
    Dự báo thời tiết 5 ngày: {weather_forecast}.
    Đề xuất: Giờ tưới tối ưu, ngày bón phân. Trả về JSON: {{"optimal_irrigation_time": "giờ", "fertilizer_day": "ngày", "reason": "lý do"}}
    """
    try:
        response = model.generate_content(prompt)
        logger.debug(f"Gemini raw response: {response.text}")

        # Clean up the response to extract only the JSON part
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()

        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi khi phân tích JSON từ Gemini: {e}. Response: '{response.text}'")
        # Return a default error structure if JSON parsing fails
        return {
            "optimal_irrigation_time": "Không xác định",
            "fertilizer_day": "Không xác định",
            "reason": f"Không thể phân tích phản hồi từ AI. Lỗi: {e}"
        }
    except Exception as e:
        logger.error(f"Lỗi không xác định từ Gemini: {e}")  
        return {
            "optimal_irrigation_time": "Không xác định",
            "fertilizer_day": "Không xác định",
            "reason": f"Lỗi khi gọi API phân tích. Lỗi: {e}"
        }

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logger.info(f"Telegram message sent at {datetime.now()}")
        else:
            logger.error(f"Failed to send: {response.text}")
    except Exception as e:
        logger.error(f"Error: {e}")

# Root endpoint để test
@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {
        "message": "FastAPI Agriculture System Running",
        "endpoints": {
            "POST /sensor": "Receive data from ESP32",
            "GET /status": "Get control status for ESP32",
            "GET /data": "Get all sensor data",
            "GET /latest": "Get latest sensor data",
            "POST /control": "Update control settings",
            "POST /telegram/webhook": "Handle Telegram commands"
        }
    }

# Lưu dữ liệu từ ESP32
@app.post("/sensor")
async def receive_sensor_data(data: SensorData):
    try:
        logger.info(f"Nhận dữ liệu từ ESP32: {data.dict()}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            
            # Lấy thông tin điều khiển từ record mới nhất
            c.execute("SELECT mode, low_threshold, high_threshold FROM sensor_data ORDER BY id DESC LIMIT 1")
            result = c.fetchone()
            
            if result:
                mode = result[0] if result[0] else "automatic"
                low_threshold = result[1] if result[1] is not None else 30
                high_threshold = result[2] if result[2] is not None else 70
            else:
                mode = "automatic"
                low_threshold = 30
                high_threshold = 70
            
            # Insert dữ liệu mới
            c.execute('''INSERT INTO sensor_data 
                         (timestamp, temperature, humidity, soil, pump_status, mode, low_threshold, high_threshold)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (timestamp, data.temperature, data.humidity, data.soil, 
                       int(data.pump_status), mode, low_threshold, high_threshold))
            conn.commit()
            conn.close()
        
        response = {
            "message": "Dữ liệu đã được lưu",
            "status": {
                "mode": mode,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "pump_status": data.pump_status
            }
        }
        logger.info(f"Đã lưu dữ liệu thành công: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu dữ liệu: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")

# Lấy dữ liệu cho dashboard
@app.get("/data")
def get_data(limit: int = 100, offset: int = 0):
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            
            # Đếm tổng số records
            c.execute("SELECT COUNT(*) FROM sensor_data")
            total = c.fetchone()[0]
            
            # Lấy records với pagination
            c.execute('''SELECT timestamp, temperature, humidity, soil, pump_status, 
                         mode, low_threshold, high_threshold 
                         FROM sensor_data ORDER BY id ASC LIMIT ? OFFSET ?''', 
                      (limit, offset))
            records = c.fetchall()
            conn.close()
        
        headers = ["Timestamp", "Temperature (°C)", "Humidity (%)", "Soil Humidity (%)", 
                   "Pump Status", "Mode", "Low Threshold (%)", "High Threshold (%)"]
        
        logger.info(f"Truy xuất {total} bản ghi cho dashboard")
        return {"headers": headers, "records": records, "total": total}
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu: {e}")
        raise HTTPException(status_code=500, detail="Lỗi server")

# Lấy dữ liệu mới nhất
@app.get("/latest")
def get_latest_data():
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''SELECT timestamp, temperature, humidity, soil, pump_status 
                         FROM sensor_data ORDER BY id DESC LIMIT 1''')
            latest_record = c.fetchone()
            conn.close()
            
            if not latest_record:
                return {"message": "Chưa có dữ liệu"}
        
        result = {
            "timestamp": latest_record[0],
            "temperature": latest_record[1],
            "humidity": latest_record[2],
            "soil": latest_record[3],
            "pump_status": bool(latest_record[4])
        }
        logger.info(f"Truy xuất dữ liệu mới nhất: {result}")
        return result
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu mới nhất: {e}")
        raise HTTPException(status_code=500, detail="Lỗi server")

# Lấy trạng thái điều khiển cho ESP32
@app.get("/status")
def get_control_status():
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''SELECT mode, low_threshold, high_threshold, pump_status 
                         FROM sensor_data ORDER BY id DESC LIMIT 1''')
            latest_record = c.fetchone()
            conn.close()
            
            if not latest_record:
                status = {
                    "mode": "automatic",
                    "low_threshold": 30,
                    "high_threshold": 70,
                    "pump_status": False,
                    "last_updated": datetime.now().isoformat()
                }
            else:
                status = {
                    "mode": latest_record[0] if latest_record[0] else "automatic",
                    "low_threshold": latest_record[1] if latest_record[1] is not None else 30,
                    "high_threshold": latest_record[2] if latest_record[2] is not None else 70,
                    "pump_status": bool(latest_record[3]) if latest_record[3] is not None else False,
                    "last_updated": datetime.now().isoformat()
                }
        
        logger.info(f"Truy xuất trạng thái điều khiển: {status}")
        return status
    except Exception as e:
        logger.error(f"Lỗi khi đọc trạng thái điều khiển: {e}")
        raise HTTPException(status_code=500, detail="Lỗi server")


@app.post("/control")
async def update_control(request: ControlRequest):
    try:
        logger.info(f"Nhận yêu cầu điều khiển: {request.dict()}")
        
       
        if request.low_threshold < 0 or request.high_threshold > 100 or request.low_threshold > 100 or request.high_threshold < 0:
            raise HTTPException(status_code=400, detail="Ngưỡng không hợp lệ (phải trong khoảng 0-100)")
        
        
        if request.low_threshold >= request.high_threshold:
            logger.warning(f"Low threshold ({request.low_threshold}) >= High threshold ({request.high_threshold})")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            
            logger.debug(f"Chèn record: timestamp={timestamp}, pump_status={int(request.pump_status)}, mode={request.mode}, "
                        f"low_threshold={request.low_threshold}, high_threshold={request.high_threshold}")
            c.execute('''INSERT INTO sensor_data 
                         (timestamp, temperature, humidity, soil, pump_status, mode, low_threshold, high_threshold)
                         VALUES (?, NULL, NULL, NULL, ?, ?, ?, ?)''',
                      (timestamp, int(request.pump_status), request.mode, 
                       request.low_threshold, request.high_threshold))
            conn.commit()
          
            c.execute("SELECT pump_status, mode FROM sensor_data ORDER BY id DESC LIMIT 1")
            last_record = c.fetchone()
            logger.debug(f"Record vừa chèn: pump_status={last_record[0]}, mode={last_record[1]}")
            conn.close()
        
        logger.info(f"Cập nhật điều khiển thành công")
        return {"message": "Cập nhật thành công", "config": request.dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật điều khiển: {e}")
        raise HTTPException(status_code=500, detail="Lỗi server")

# Webhook để nhận lệnh từ Telegram
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received webhook data: {data}")
        if "message" in data and "text" in data["message"]:
            command = data["message"]["text"]
            chat_id = data["message"]["chat"]["id"]
            logger.info(f"Processing command: {command}, chat_id: {chat_id}")
            if command == "/analyst":
                with db_lock:
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1")
                    row = c.fetchone()
                    conn.close()
                    logger.info(f"Database row: {row}")

                if row:
                    temperature, humidity, soil = row[2], row[3], row[4]
                    logger.info(f"Sensor data: temp={temperature}, hum={humidity}, soil={soil}")
                    forecast = await get_weather_forecast()
                    logger.info(f"Weather forecast: {forecast}")
                    analysis = await analyze_irrigation_and_fertilizer({"temperature": temperature, "humidity": humidity, "soil": soil}, forecast)
                    logger.info(f"Gemini analysis: {analysis}")
                    message = f"*Phân tích tưới tiêu*\n- Độ ẩm đất: {soil}%\n- Nhiệt độ: {temperature}°C\n- Độ ẩm không khí: {humidity}%\n- Giờ tưới tối ưu: {analysis['optimal_irrigation_time']}\n- Ngày bón phân: {analysis['fertilizer_day']}\n- Lý do: {analysis['reason']}"
                    logger.info(f"Sending message: {message}") 
                    send_telegram_message(message)
                else:
                    message = "Không có dữ liệu cảm biến gần đây."
                    logger.info(f"Sending message: {message}")
                    send_telegram_message(message)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Lỗi khi xử lý webhook Telegram: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi server")

# Thiết lập webhook khi khởi động
@app.on_event("startup")
async def on_startup():
    models = genai.list_models()
    logger.info(f"Available Gemini models: {[m.name for m in models]}")
    webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url=https://agricultural-backend.onrender.com/telegram/webhook"
    response = requests.get(webhook_url)
    logger.info(f"Webhook setup: {response.text}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)