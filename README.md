<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/PhoBERT-NLP-00D4AA?style=for-the-badge&logo=huggingface&logoColor=white" alt="PhoBERT"/>
  <img src="https://img.shields.io/badge/Gemini-AI-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini"/>
</p>

<h1 align="center">📊📊 Analyzing Effects of News on Stock Prices using NLP and Econometric Models
Evidence from the Vietnamese Stock Market</h1>

<p align="center">
  <strong>Phân tích ảnh hưởng của tin tức đến giá cổ phiếu Việt Nam</strong><br>
  <em>Sử dụng PhoBERT, AI Chatbot và các mô hình kinh tế lượng</em>
</p>

<p align="center">
  <a href="#-tính-năng">Tính năng</a> •
  <a href="#-cài-đặt-nhanh">Cài đặt</a> •
  <a href="#-hướng-dẫn-sử-dụng">Hướng dẫn</a> •
  <a href="#-công-nghệ">Công nghệ</a>
</p>

---

## 🎯 Giới thiệu

Dự án này nghiên cứu mối quan hệ giữa tâm lý thị trường tin tức tài chính và biến động giá cổ phiếu trên thị trường chứng khoán Việt Nam.
Sử dụng PhoBERT (công nghệ xử lý ngôn ngữ tự nhiên tiếng Việt) để trích xuất cảm xúc và các mô hình kinh tế lượng (hệ số tương quan Pearson, quan hệ nhân quả Granger và VAR ngưỡng), nghiên cứu đánh giá liệu các tín hiệu cảm xúc có thể giải thích hoặc dự đoán biến động giá cổ phiếu, đặc biệt là đối với các cổ phiếu có khả năng bị thao túng.
- 🤖 **AI Chatbot** với Google Gemini - Trả lời real-time về thị trường
- 📊 **8 loại biểu đồ** chuyên nghiệp giống FireAnt
- 📈 **PhoBERT** - Phân tích cảm xúc tin tức tiếng Việt
- 📉 **Mô hình kinh tế lượng** - Pearson, Granger, TVAR

---

## ✨ Tính năng

### 🤖 AI Chatbot Thông minh
| Tính năng | Mô tả |
|-----------|-------|
| 💬 Real-time Chat | Trả lời câu hỏi về cổ phiếu, thị trường |
| 📊 Phân tích kỹ thuật | RSI, SMA, Golden Cross tự động |
| 📰 Sentiment Analysis | Phân tích tin tức & tác động giá |
| 💾 Lưu lịch sử | Auto-save theo session |

### 📈 Biểu đồ Chuyên nghiệp

```
🕯️ Candle    📈 Line      📊 Bar       🔲 Step
🏔️ Mountain  🌊 Wave      ⚫ Scatter   📉 Histogram
```

**FireAnt-style Interactions:**
- ✅ Smooth transitions 300ms
- ✅ Pan mode (kéo để di chuyển)
- ✅ Crosshair spike lines
- ✅ Range selector: 1T → 9T → 5N → All

### 📊 Chỉ báo Kỹ thuật (15+)

| Nhóm | Chỉ báo |
|------|---------|
| **Trend** | SMA, EMA, MACD, ADX |
| **Momentum** | RSI, Stochastic |
| **Volatility** | Bollinger Bands, ATR |
| **Volume** | OBV, VWAP |
| **Support/Resistance** | Fibonacci Retracement |

### 📉 Mô hình Kinh tế lượng

| Mô hình | Ứng dụng |
|---------|----------|
| **Pearson** | Đo tương quan sentiment-giá |
| **Granger** | Kiểm định nhân quả thời gian |
| **TVAR** | Dự báo theo regime thị trường |

---

## 🚀 Cài đặt Nhanh

### Yêu cầu
- Python 3.8+
- RAM 4GB+ (khuyến nghị 8GB)
- Disk 2GB+

### Bước 1: Clone & Setup

```bash
# Clone repository
git clone https://github.com/your-username/Stock_News_Project.git
cd Stock_News_Project

# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Hoặc Windows CMD
.\venv\Scripts\activate.bat

# Hoặc Linux/Mac
source venv/bin/activate
```

### Bước 2: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Cấu hình API Keys

Tạo file `config/settings.py`:

```python
# Google Gemini API (bắt buộc cho Chatbot)
GEMINI_API_KEY = "your_gemini_api_key"

# Lấy key tại: https://makersuite.google.com/app/apikey
```

### Bước 4: Chạy ứng dụng

```bash
streamlit run app.py
```

🌐 Mở browser: **http://localhost:8501**

---

## 📖 Hướng dẫn Sử dụng

### 🎛️ Sidebar Controls

```
📊 Select Ticker  → Chọn mã cổ phiếu (VCB, FLC, ...)
📈 Chart Type     → 8 loại biểu đồ
📰 Data Type      → Content / Title
⏳ Time Period    → Before / After Scandal
```

### 🗂️ Tabs chính

| Tab | Chức năng |
|-----|-----------|
| **Pricing** | Biểu đồ giá, chỉ báo kỹ thuật |
| **Sentiment** | Phân tích cảm xúc PhoBERT |
| **News** | Tin tức & sentiment scores |
| **Pearson** | Correlation analysis |
| **Granger** | Causality testing |
| **TVAR** | Threshold VAR model |

### 🤖 AI Chatbot

1. Click nút **🤖** ở góc phải dưới
2. Nhập câu hỏi: *"Phân tích VCB?"*
3. Bot trả lời với dữ liệu real-time

**Ví dụ câu hỏi:**
- "RSI của VCB là bao nhiêu?"
- "So sánh VCB với BID"
- "Tin tức VCB ảnh hưởng giá thế nào?"

---

## 📁 Cấu trúc Dự án

```
Stock_News_Project/
├── 📄 app.py                 # Entry point
├── 📄 requirements.txt       # Dependencies
│
├── 📂 config/                # Cấu hình
│   └── settings.py           # API keys
│
├── 📂 models/                # AI & ML Models
│   ├── sentiment_phobert.py  # PhoBERT sentiment
│   ├── chatbot_services.py   # Gemini chatbot
│   ├── granger_test.py       # Granger causality
│   └── tvar_model.py         # TVAR model
│
├── 📂 ui/                    # Streamlit UI
│   ├── overview_tab.py       # Tab tổng quan
│   ├── sentiment_tab.py      # Tab sentiment
│   ├── chatbot_ui.py         # Chatbot dialog
│   └── ...
│
├── 📂 utils/                 # Utilities
│   ├── charts.py             # Chart rendering
│   ├── indicators.py         # Technical indicators
│   ├── vndirect_api.py       # Real-time API
│   └── data_loader.py        # Data loading
│
└── 📂 data/                  # Datasets
    ├── prices/               # Historical prices
    └── vnecon_*/             # News data
```

---

## 🛠️ Công nghệ

<table>
<tr>
<td align="center"><strong>Frontend</strong></td>
<td align="center"><strong>AI/ML</strong></td>
<td align="center"><strong>Data</strong></td>
</tr>
<tr>
<td>
  
- Streamlit
- Plotly
- Matplotlib

</td>
<td>

- PhoBERT (VinAI)
- Google Gemini
- Transformers

</td>
<td>

- Pandas
- Vnstock API
- VNDirect API

</td>
</tr>
</table>

---

## 🐛 Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| Import PhoBERT fail | `pip install transformers torch --upgrade` |
| API key error | Kiểm tra `config/settings.py` |
| Cache issues | `python clear_cache.py` |
| Missing data | Kiểm tra thư mục `data/` |

---

## 📊 Mã Cổ phiếu Hỗ trợ

| Nhóm | Mã |
|------|-----|
| **FLC Group** | FLC, GAB, HAI, AMD, ART |
| **VN30** | VCB, BID, CTG, TCB, MBB, VPB, FPT, HPG, VNM, ... |
| **Custom** | Nhập bất kỳ mã nào |

---

## 📈 Kết quả Nghiên cứu

> Dự án phân tích ảnh hưởng tin tức đến giá cổ phiếu trước/sau các scandal tài chính lớn tại Việt Nam.

**Phát hiện chính:**
- 📊 Sentiment score tương quan đáng kể với biến động giá
- 📉 Tin tức tiêu cực tác động mạnh hơn tin tích cực
- 📈 TVAR cải thiện dự báo trong thị trường bất ổn

---

## 🤝 Đóng góp

```bash
# 1. Fork project
# 2. Tạo feature branch
git checkout -b feature/AmazingFeature

# 3. Commit changes
git commit -m 'Add AmazingFeature'

# 4. Push & Create PR
git push origin feature/AmazingFeature
```
---

## 🙏 Credits

- **PhoBERT**: [VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- **Vnstock**: [Thinh Vu](https://github.com/thinh-vu/vnstock)
- **Streamlit**: [Streamlit Community](https://streamlit.io)
- **Google Gemini**: [Google AI](https://ai.google.dev)

---

<p align="center">
  <strong>⭐ Nếu dự án hữu ích, hãy cho một star nhé!</strong>
</p>

<<<<<<< HEAD
<p align="center">
  📧 Có thắc mắc? Tạo <a href="../../issues">Issue</a> trên GitHub
</p>
=======
**Nguyễn Hoàng Đồng**

---

## 🙏 Acknowledgments

- PhoBERT model: VinAI Research
- Vnstock library: Thinh Vu
- Google Gemini API
- Streamlit Community

---
**⭐ Nếu dự án hữu ích, hãy cho một star nhé!**
