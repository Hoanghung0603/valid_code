import os
from dotenv import load_dotenv
from src.utils.proxy import parse_proxy_string

# Tải biến môi trường từ file .env
load_dotenv(override=True)

class Settings:
    """Lớp quản lý tập trung các cấu hình và biến môi trường (Environment Variables)"""
    
    # LLM API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_URL = os.getenv("GROQ_URL") # Để None để SDK tự lấy mặc định (https://api.groq.com)

    # Proxies cào dữ liệu: Chỉ giữ lại 2 key PROXY_240S và PROXY_15S
    PROXY_240S = parse_proxy_string(os.getenv("PROXY_240S", ""))
    PROXY_15S = parse_proxy_string(os.getenv("PROXY_15S", ""))

# Khởi tạo một instance duy nhất (Singleton pattern) để import ở mọi nơi
settings = Settings()
