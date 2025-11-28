import streamlit as st
from config.settings import configure_logging
import warnings
import sys
import os

# Suppress all warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Tắt warnings của Google/GRPC/ALTS
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

# Tắt TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Torch settings
if "TORCH_LOGS" in os.environ:
    del os.environ["TORCH_LOGS"]

# Suppress stderr temporarily for torch imports
import io
_original_stderr = sys.stderr
sys.stderr = io.StringIO()

import logging
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("torch._classes").setLevel(logging.CRITICAL)
logging.getLogger("torch.classes").setLevel(logging.CRITICAL)
logging.getLogger("absl").setLevel(logging.CRITICAL)
logging.getLogger("google").setLevel(logging.CRITICAL)

# Restore stderr
sys.stderr = _original_stderr

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="News Sentiment & Stock Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hiển thị loading indicator ngay lập tức
placeholder = st.empty()
with placeholder.container():
    st.markdown("""
        <div style='text-align:center; padding:100px;'>
            <h2 style='color:#22c55e;'>⚡ Đang tải ứng dụng...</h2>
            <p style='color:#94a3b8;'>Vui lòng chờ trong giây lát</p>
        </div>
    """, unsafe_allow_html=True)

# ==============================
# LOGGING
# ==============================
configure_logging()
logger = logging.getLogger(__name__)

# ==============================
# SAFE IMPORT FOR TABS
# ==============================
def safe_import(module_name, render_name="render"):
    """Safely import modules with error handling"""
    try:
        module = __import__(module_name, fromlist=[render_name])
        return getattr(module, render_name)
    except Exception as e:
        st.error(f"⚠️ Không thể tải module `{module_name}`: {e}")
        logger.error(e)
        return lambda *args, **kwargs: st.warning(f"Module `{module_name}` không khả dụng.")

# Lazy import các tab - chỉ load khi cần
@st.cache_resource(show_spinner=False)
def get_tab_module(module_name):
    """Lazy load từng tab module khi được gọi"""
    return safe_import(module_name)

# LAZY LOAD CHATBOT - chỉ import khi user click button
@st.cache_resource(show_spinner=False)
def get_chatbot_functions():
    """Lazy load chatbot functions chỉ khi cần"""
    try:
        from ui.chatbot_ui import render_floating_button, show_popup_dialog
        return render_floating_button, show_popup_dialog
    except Exception as e:
        logger.error(f"Cannot import chatbot: {e}")
        return lambda: st.warning("Chatbot không khả dụng"), lambda: None

# ==============================
# SIDEBAR CONFIGURATION
# ==============================
st.sidebar.title("📊 Data Configuration")

# Chọn nhóm cổ phiếu
category = st.sidebar.selectbox(
    "Stock Category:",
    ["FLC Group Stocks", "VN30 Stocks", "Custom Group"],
    key="sidebar_category"
)

tickers = {
    "FLC Group Stocks": ["FLC", "GAB", "HAI", "AMD", "ART"],
    "VN30 Stocks": [
        "VCB", "BID", "CTG", "TCB", "MBB", "VPB", "VHM", "VIC", "VNM", "FPT",
        "HPG", "MSN", "SAB", "VRE", "SSI", "STB", "SHB", "GAS", "BVH", "NVL"
    ],
}

# Chọn mã cổ phiếu
ticker = st.sidebar.selectbox(
    "Select Ticker:",
    tickers.get(category, []),
    index=0 if tickers.get(category) else None,
    key="sidebar_ticker"
)

# Cho phép nhập custom ticker
custom_ticker = st.sidebar.text_input(
    "Or enter a custom ticker (optional):",
    key="sidebar_custom_ticker"
)
if custom_ticker.strip():
    ticker = custom_ticker.strip().upper()

# Chọn loại dữ liệu
st.sidebar.markdown("### 📰 Data Type")
data_type = st.sidebar.radio(
    "Select data source:",
    ["Content", "Title"],
    index=0,
    horizontal=True,
    key="sidebar_data_type"
)

# Chọn giai đoạn
st.sidebar.markdown("### ⏳ Time Period")
time_period = st.sidebar.radio(
    "Select dataset period:",
    ["Before Scandal", "After Scandal"],
    index=0,
    horizontal=True,
    key="sidebar_time_period"
)

# ==============================
# 📊 TECHNICAL INDICATORS SETTINGS
# ==============================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Chỉ số kỹ thuật")

with st.sidebar.expander("⚙️ Cấu hình biểu đồ", expanded=False):
    st.selectbox("Loại biểu đồ", ["Candle", "Line"], index=0, key="chart_type")
    st.checkbox("Hiển thị Volume", value=True, key="show_volume")

with st.sidebar.expander("📈 Moving Averages", expanded=False):
    st.checkbox("SMA 20", value=True, key="show_sma20")
    st.checkbox("SMA 50", value=True, key="show_sma50")
    st.checkbox("EMA 12", value=False, key="show_ema12")
    st.checkbox("EMA 26", value=False, key="show_ema26")

with st.sidebar.expander("📉 Oscillators", expanded=False):
    st.checkbox("RSI", value=False, key="show_rsi")
    st.checkbox("MACD", value=False, key="show_macd")
    st.checkbox("Stochastic", value=False, key="show_stoch")

with st.sidebar.expander("📊 Trend & Volatility", expanded=False):
    st.checkbox("Bollinger Bands", value=False, key="show_bb")
    st.checkbox("ADX", value=False, key="show_adx")
    st.checkbox("ATR", value=False, key="show_atr")

with st.sidebar.expander("📦 Volume Indicators", expanded=False):
    st.checkbox("OBV", value=False, key="show_obv")
    st.checkbox("VWAP", value=False, key="show_vwap")

# Thông tin hướng dẫn
st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Selected Ticker:** `{ticker}`  
    **Data Type:** `{data_type}`  
    **Period:** `{time_period}`

    💡 Dữ liệu được tự động tải từ thư mục tương ứng:
    `data/vnecon_{data_type.lower()}_{'before' if time_period == 'Before Scandal' else 'after'}_scandals/`
    """
)

# Lưu cấu hình vào session_state
st.session_state["ticker"] = ticker
st.session_state["data_type"] = data_type
st.session_state["time_period"] = time_period

# Xóa loading indicator
placeholder.empty()

# ==============================
# MAIN HEADER
# ==============================
st.markdown(
    """
    <h2 style='text-align:center; color:#22c55e;'>📗 Stock Sentiment & Econometric Dashboard</h2>
    <p style='text-align:center; color:#94a3b8;'>
    Analyze the impact of news sentiment on stock price dynamics using <b>PhoBERT</b> and econometric models.
    </p>
    """,
    unsafe_allow_html=True,
)

# ==============================
# MAIN TAB NAVIGATION
# ==============================
tabs = st.tabs([
    "📘 Pricing Tab",
    "💬 Phân tích cảm xúc (PhoBERT)",
    "📊 Kiểm định Tương quan (Pearson)",
    "🔁 Kiểm định Nhân quả (Granger)",
    "📉 Mô hình Ngưỡng (TVAR)",
    "📰 News Articles"
])

with tabs[0]:
    overview_tab = get_tab_module("ui.overview_tab")
    overview_tab(ticker)
with tabs[1]:
    sentiment_tab = get_tab_module("ui.sentiment_tab")
    sentiment_tab(ticker)
with tabs[2]:
    pearson_tab = get_tab_module("ui.pearson_tab")
    pearson_tab(ticker)
with tabs[3]:
    granger_tab = get_tab_module("ui.granger_tab")
    granger_tab(ticker)
with tabs[4]:
    tvar_tab = get_tab_module("ui.tvar_tab")
    tvar_tab(ticker)
with tabs[5]:
    news_tab = get_tab_module("ui.news_tab")
    news_tab(ticker)


# ==============================
# FLOATING CHATBOT BUTTON (LAZY LOADED)
# ==============================
render_floating_button, show_popup_dialog = get_chatbot_functions()
render_floating_button()

# ==============================
# CHATBOT POPUP DIALOG
# ==============================
show_popup_dialog()

# ==============================
# FOOTER
# ==============================
st.markdown("""
---
<center style='color:gray; font-size:13px'>
FinTech Research | PhoBERT × Econometrics × Streamlit Dashboard © 2025
</center>
""", unsafe_allow_html=True)