# 📊 Stock News Sentiment & Econometric Analysis Dashboard

Web application Streamlit phân tích ảnh hưởng của tin tức đến giá cổ phiếu Việt Nam sử dụng **PhoBERT**, **AI Chatbot** và các mô hình kinh tế lượng (Pearson, Granger, TVAR).

---

## 🎯 Mô tả dự án

Dự án này phát triển một hệ thống phân tích toàn diện để nghiên cứu mối quan hệ giữa tin tức kinh tế và biến động giá cổ phiếu trên thị trường chứng khoán Việt Nam. Hệ thống sử dụng kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) với mô hình PhoBERT để phân tích cảm xúc tin tức, kết hợp với các phương pháp kinh tế lượng để đo lường tác động của tin tức đến giá cổ phiếu.

### 🎓 Mục tiêu nghiên cứu

- Phân tích cảm xúc tin tức tài chính tiếng Việt bằng PhoBERT
- Đo lường tương quan giữa sentiment score và biến động giá cổ phiếu
- Kiểm định mối quan hệ nhân quả Granger giữa tin tức và giá
- Xây dựng mô hình TVAR để dự báo giá theo chế độ thị trường
- Cung cấp công cụ trực quan hóa và phân tích tương tác

---

## ✨ Tính năng nổi bật

### 🤖 AI Chatbot
- Trợ lý thông minh tích hợp Google Gemini API
- Tự động lưu và quản lý lịch sử hội thoại theo session
- Hỗ trợ nhiều phiên chat khác nhau
- Giao diện thân thiện, dễ sử dụng

### 📈 Phân tích cảm xúc (Sentiment Analysis)
- Sử dụng mô hình PhoBERT được fine-tune cho tiếng Việt
- Phân loại tin tức: Tích cực, Tiêu cực, Trung tính
- Tính toán sentiment score và phân tích theo thời gian
- Hỗ trợ phân tích cả tiêu đề và nội dung tin tức

### 📊 Kiểm định thống kê
- **Pearson Correlation**: Đo lường mối tương quan tuyến tính
- **Granger Causality Test**: Kiểm định nhân quả theo thời gian
- **TVAR Model**: Mô hình Vector Autoregression với ngưỡng (Threshold VAR)

### 💹 Phân tích kỹ thuật
- **Dữ liệu realtime**: Tích hợp VNDirect API
- **Dữ liệu lịch sử**: Vnstock API
- **Biểu đồ nến**: Candlestick charts chuyên nghiệp
- **15+ chỉ báo kỹ thuật**:
  - Moving Averages: SMA, EMA, WMA
  - Momentum: RSI, Stochastic Oscillator
  - Trend: MACD, ADX, Parabolic SAR
  - Volatility: Bollinger Bands, ATR
  - Volume: OBV, VWAP, Volume Profile
  - Patterns: Candlestick patterns, Chart patterns

### 📉 Trực quan hóa dữ liệu
- Interactive charts với Plotly
- Biểu đồ so sánh sentiment và giá cổ phiếu
- Visualization cho kết quả kiểm định thống kê
- Dashboard tổng quan thị trường

---

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Windows/Linux/MacOS
- RAM: 4GB+ (khuyến nghị 8GB)
- Disk: 2GB+ dung lượng trống

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd Stock_News_Project
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)

```bash
python -m venv venv
```

Kích hoạt môi trường ảo:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình API Keys

Tạo file `config/settings.py` và thêm các API keys:

```python
# Google Gemini API
GOOGLE_API_KEY = "your_google_gemini_api_key"

# VNDirect API (nếu cần)
VNDIRECT_API_KEY = "your_vndirect_api_key"
```

**Lấy API keys:**
- Google Gemini: https://makersuite.google.com/app/apikey
- VNDirect: https://www.vndirect.com.vn/

### Bước 5: Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

---

## 📁 Cấu trúc dự án

```
Stock_News_Project/
├── app.py                          # Entry point chính của ứng dụng
├── requirements.txt                # Python dependencies
├── README.md                       # Tài liệu dự án
├── clear_cache.py                  # Script xóa cache
├── test_endpoints.py               # Test API endpoints
│
├── config/                         # Cấu hình
│   ├── settings.py                 # API keys và settings
│   └── cache_config.py             # Cấu hình cache
│
├── models/                         # Models và thuật toán
│   ├── sentiment_phobert.py        # PhoBERT sentiment analysis
│   ├── pearson_test.py             # Pearson correlation test
│   ├── granger_test.py             # Granger causality test
│   ├── tvar_model.py               # TVAR model implementation
│   └── chatbot_services.py         # Chatbot logic với Gemini
│
├── ui/                             # Giao diện Streamlit
│   ├── chatbot_ui.py               # Chatbot interface
│   ├── sentiment_tab.py            # Tab phân tích sentiment
│   ├── pearson_tab.py              # Tab Pearson correlation
│   ├── granger_tab.py              # Tab Granger causality
│   ├── tvar_tab.py                 # Tab TVAR model
│   ├── news_tab.py                 # Tab quản lý tin tức
│   └── overview_tab.py             # Tab tổng quan
│
├── utils/                          # Utilities
│   ├── data_loader.py              # Load dữ liệu từ files
│   ├── charts.py                   # Vẽ biểu đồ
│   ├── indicators.py               # Tính chỉ báo kỹ thuật
│   ├── patterns.py                 # Nhận diện patterns
│   ├── visualization.py            # Visualization functions
│   ├── vndirect_api.py             # VNDirect API integration
│   └── chat_history_manager.py     # Quản lý lịch sử chat
│
└── data/                           # Dữ liệu
    ├── prices/                     # Giá cổ phiếu lịch sử
    ├── chat_history/               # Lịch sử chat sessions
    ├── vnecon_before_scandals/     # Tin tức trước scandal
    ├── vnecon_after_scandals/      # Tin tức sau scandal
    ├── vnecon_title_before_scandals/
    └── vnecon_title_after_scandals/
```

---

## 📖 Hướng dẫn sử dụng

### 1. AI Chatbot
- Chọn session từ sidebar hoặc tạo session mới
- Nhập câu hỏi về thị trường, cổ phiếu, tin tức
- Lịch sử hội thoại được lưu tự động
- Có thể xóa lịch sử hoặc chuyển session

### 2. Phân tích Sentiment
- Chọn mã cổ phiếu cần phân tích
- Chọn khoảng thời gian (before/after scandal)
- Xem kết quả phân tích: sentiment score, phân phối, correlation với giá
- Xuất báo cáo và biểu đồ

### 3. Kiểm định Pearson
- Chọn cổ phiếu và khoảng thời gian
- Hệ thống tự động tính correlation giữa sentiment và các metrics giá
- Hiển thị heatmap và các chỉ số thống kê

### 4. Kiểm định Granger
- Chọn biến độc lập và phụ thuộc
- Đặt max lag để test
- Xem kết quả F-statistic và p-value
- Diễn giải mối quan hệ nhân quả

### 5. Mô hình TVAR
- Chọn biến và threshold
- Huấn luyện mô hình với dữ liệu lịch sử
- Xem kết quả dự báo theo regime
- Đánh giá độ chính xác mô hình

---

## 🔧 Cấu hình nâng cao

### Cache Management

Clear cache khi cần:
```bash
python clear_cache.py
```

Hoặc từ Streamlit UI: Menu > Clear Cache

### Custom Settings

Chỉnh sửa `config/settings.py` để tùy chỉnh:
- API keys
- Model parameters
- Data paths
- Cache settings

---

## 📊 Dữ liệu

### Mã cổ phiếu hỗ trợ

- **BID**: BIDV
- **CTG**: VietinBank
- **VCB**: Vietcombank
- **FLC**: FLC Group
- **GAB**: Ngân hàng Quân Đội
- **HAI**: Hải Phát Invest
- **SHB**: SHB Bank

### Nguồn dữ liệu

- **Tin tức**: VnEconomy (crawled data)
- **Giá cổ phiếu lịch sử**: Vnstock API
- **Giá realtime**: VNDirect API

---

## 🛠️ Công nghệ sử dụng

### Frontend & UI
- **Streamlit**: Web framework
- **Plotly**: Interactive charts
- **Matplotlib/Seaborn**: Static visualizations

### Machine Learning & NLP
- **PhoBERT**: Sentiment analysis (vinai/phobert-base)
- **Transformers**: Hugging Face library
- **Google Gemini**: AI chatbot

### Data Analysis
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Statsmodels**: Statistical tests
- **SciPy**: Scientific computing

### APIs & Data
- **Vnstock**: Historical stock data
- **VNDirect**: Realtime market data
- **Google Gemini API**: AI assistant

---

## 📝 Dependencies chính

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
transformers>=4.30.0
torch>=2.0.0
statsmodels>=0.14.0
scipy>=1.10.0
vnstock>=1.0.0
google-generativeai>=0.3.0
```

Xem đầy đủ trong `requirements.txt`

---

## 🐛 Troubleshooting

### Lỗi import PhoBERT
```bash
pip install transformers torch --upgrade
```

### Lỗi API key
- Kiểm tra file `config/settings.py`
- Đảm bảo API keys hợp lệ và còn quota

### Lỗi cache
```bash
python clear_cache.py
```

### Lỗi missing data
- Kiểm tra folder `data/` có đầy đủ files
- Download dữ liệu từ nguồn nếu thiếu

---

## 📈 Kết quả nghiên cứu

Dự án đã phân tích ảnh hưởng của tin tức đến giá cổ phiếu trước và sau các scandal tài chính lớn tại Việt Nam, bao gồm:
- FLC Group scandal
- Ngân hàng Đông Á (GAB)
- Hải Phát Invest (HAI)

Kết quả cho thấy:
- Sentiment score có tương quan đáng kể với biến động giá
- Tin tức tiêu cực có tác động mạnh hơn tin tích cực
- Mô hình TVAR cải thiện độ chính xác dự báo trong điều kiện thị trường bất ổn

---

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! 

1. Fork project
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📄 License

Dự án này được phát triển cho mục đích nghiên cứu và giáo dục.

---

## 👥 Tác giả

**Stock News Project Team**

---

## 📧 Liên hệ

Nếu có thắc mắc hoặc góp ý, vui lòng tạo Issue trên GitHub.

---

## 🙏 Acknowledgments

- PhoBERT model: VinAI Research
- Vnstock library: Thinh Vu
- Google Gemini API
- Streamlit Community

---

**⭐ Nếu dự án hữu ích, hãy cho một star nhé!**
