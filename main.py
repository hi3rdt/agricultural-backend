from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import logging
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock

# Cấu hình logging chi tiết hơn
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fastapi.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Cho phép Next.js và ESP32 truy cập API
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

# Khởi tạo database nếu chưa có
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

# Model dữ liệu từ cảm biến ESP32
class SensorData(BaseModel):
    temperature: float
    humidity: float
    soil: float
    pump_status: bool

# Model dữ liệu điều khiển từ dashboard
class ControlRequest(BaseModel):
    mode: str
    low_threshold: int
    high_threshold: int
    pump_status: bool

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
            "POST /control": "Update control settings"
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
        
        logger.info(f"📋 Truy xuất {total} bản ghi cho dashboard")
        return {"headers": headers, "records": records, "total": total}
    except Exception as e:
        logger.error(f"❌ Lỗi khi đọc dữ liệu: {e}")
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

# Cập nhật điều khiển từ dashboard
@app.post("/control")
async def update_control(request: ControlRequest):
    try:
        logger.info(f"🎮 Nhận yêu cầu điều khiển: {request.dict()}")
        
        # Kiểm tra ngưỡng hợp lệ
        if request.low_threshold < 0 or request.high_threshold > 100 or request.low_threshold > 100 or request.high_threshold < 0:
            raise HTTPException(status_code=400, detail="Ngưỡng không hợp lệ (phải trong khoảng 0-100)")
        
        # Warning nếu low >= high nhưng vẫn cho phép
        if request.low_threshold >= request.high_threshold:
            logger.warning(f"Low threshold ({request.low_threshold}) >= High threshold ({request.high_threshold})")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            # Thêm log để kiểm tra giá trị trước khi chèn
            logger.debug(f"Chèn record: timestamp={timestamp}, pump_status={int(request.pump_status)}, mode={request.mode}, "
                        f"low_threshold={request.low_threshold}, high_threshold={request.high_threshold}")
            c.execute('''INSERT INTO sensor_data 
                         (timestamp, temperature, humidity, soil, pump_status, mode, low_threshold, high_threshold)
                         VALUES (?, NULL, NULL, NULL, ?, ?, ?, ?)''',
                      (timestamp, int(request.pump_status), request.mode, 
                       request.low_threshold, request.high_threshold))
            conn.commit()
            # Kiểm tra lại record vừa chèn
            c.execute("SELECT pump_status, mode FROM sensor_data ORDER BY id DESC LIMIT 1")
            last_record = c.fetchone()
            logger.debug(f"Record vừa chèn: pump_status={last_record[0]}, mode={last_record[1]}")
            conn.close()
        
        logger.info(f"✅ Cập nhật điều khiển thành công")
        return {"message": "Cập nhật thành công", "config": request.dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi khi cập nhật điều khiển: {e}")
        raise HTTPException(status_code=500, detail="Lỗi server")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)