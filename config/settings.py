import logging

def configure_logging():
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

THEME = {
    "bg": "#0f172a",
    "text": "#e2e8f0",
    "accent": "#38bdf8"
}

# Cấu hình Google Gemini API cho chatbot
# Lấy API key miễn phí tại: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyDvAd-GA5XUiJq3foKN706m7W3GifEACJI"  # Thay bằng API key của bạn
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # hoặc "gemini-1.5-pro"

import logging
logging.getLogger("torch").setLevel(logging.ERROR)
