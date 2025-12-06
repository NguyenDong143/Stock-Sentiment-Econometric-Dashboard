from datetime import datetime
import os
import pandas as pd
import streamlit as st
import logging
# import investpy # ĐÃ BỊ LOẠI BỎ
import yfinance as yf 
from typing import Optional, Dict
from utils.vndirect_api import get_vndirect_api 

# 🆕 VNSTOCK - Lazy loading để tránh lỗi circular import
_vnstock_module = None

def _get_vnstock():
    """Lazy load vnstock module to avoid circular import issues with vnai."""
    global _vnstock_module
    if _vnstock_module is None:
        try:
            import warnings
            warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
            from vnstock import Vnstock
            _vnstock_module = Vnstock
        except Exception as e:
            logger.error(f"Failed to import vnstock: {e}")
            _vnstock_module = False  # Mark as failed to avoid retrying
    return _vnstock_module if _vnstock_module is not False else None


logger = logging.getLogger(__name__)


# ======================================================
# 🔧 HÀM ĐỌC FILE EXCEL AN TOÀN & CHUẨN HÓA DỮ LIỆU
# ======================================================
@st.cache_data(show_spinner=False)
def _safe_load_excel(path: str) -> pd.DataFrame:
    """Đọc file Excel an toàn, chuẩn hóa tên cột, kiểu dữ liệu và tránh lỗi Arrow."""
    if not os.path.exists(path):
        st.warning(f"⚠️ Không tìm thấy file: `{path}`")
        logger.warning(f"File không tồn tại: {path}")
        return pd.DataFrame()

    try:
        # Sử dụng engine "openpyxl" là tiêu chuẩn cho Streamlit
        df = pd.read_excel(path, engine="openpyxl") 
    except Exception as e:
        st.error(f"❌ Lỗi đọc file `{path}`: {e}")
        logger.error(f"Lỗi đọc file {path}: {e}")
        return pd.DataFrame()

    # 🔹 Chuẩn hóa tên cột — chữ thường, loại bỏ khoảng trắng thừa
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 🔹 Chuẩn hóa cột 'date' nếu có
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").dropna(subset=["date"])

    # 🔹 Làm sạch dữ liệu số: thay ',' bằng '.', xóa '%', và ép kiểu an toàn
    for col in df.columns:
        if df[col].dtype == "object":
            # Loại bỏ các ký tự không phải số và chuẩn hóa dấu thập phân
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False) # Xóa dấu phẩy phân cách hàng nghìn
                .str.replace(".", "", regex=False) # Sẽ thay thế lại nếu cần
                .str.replace("%", "", regex=False)
                .str.replace("₫", "", regex=False)
                .str.replace("vnđ", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            # Thử ép kiểu số
            try:
                # Ép kiểu an toàn, NaN nếu thất bại
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    # 🔹 Ép các cột không phải số về string (fix lỗi Arrow / Streamlit caching)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    # 🔹 Kiểm tra bắt buộc các cột cảm xúc (nếu có)
    required_cols = ["tích cực", "tiêu cực", "trung tính"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Thiếu các cột {missing} trong file {path}")

    return df


# ======================================================
# 📰 TẢI DỮ LIỆU CẢM XÚC THEO CẤU HÌNH SIDEBAR
# ======================================================
@st.cache_data(show_spinner=False, ttl=7200)
def load_sentiment_data(
    ticker: Optional[str] = None, data_type: str = "Content", time_period: str = "Before Scandal"
) -> pd.DataFrame:
    """
    Tải dữ liệu cảm xúc dựa trên cấu hình được chọn trong sidebar.
    (Giữ nguyên)
    """
    base_dir = "data"
    type_map = {"Content": "vnecon", "Title": "vnecon_title"}
    period_map = {"Before Scandal": "before_scandals", "After Scandal": "after_scandals"}

    folder_name = f"{type_map.get(data_type, 'vnecon')}_{period_map.get(time_period, 'before_scandals')}"
    data_dir = os.path.join(base_dir, folder_name)

    if not os.path.exists(data_dir):
        st.error(f"❌ Thư mục dữ liệu không tồn tại: `{data_dir}`")
        return pd.DataFrame()

    # Nếu người dùng chọn mã cụ thể
    if ticker:
        file_path = os.path.join(data_dir, f"{ticker.upper()}.xlsx")
        if os.path.exists(file_path):
            df = _safe_load_excel(file_path)
            return df
        else:
            st.warning(f"⚠️ Không tìm thấy file `{ticker}.xlsx` trong `{folder_name}/`.")
            return pd.DataFrame()

    # Nếu không có ticker -> hợp nhất toàn bộ file trong thư mục
    dfs = []
    for file in os.listdir(data_dir):
        if file.endswith((".xlsx", ".xls")) and not file.startswith("~$"):
            df = _safe_load_excel(os.path.join(data_dir, file))
            if not df.empty:
                df["ticker"] = file.replace(".xlsx", "").replace(".xls", "").upper()
                dfs.append(df)

    if not dfs:
        st.error(f"❌ Không tìm thấy file Excel nào trong `{folder_name}/`.")
        return pd.DataFrame()

    st.info(f"📘 Đã hợp nhất dữ liệu trong `{folder_name}/` ({len(dfs)} file).")
    return pd.concat(dfs, ignore_index=True)


# ======================================================
# 💹 TẢI DỮ LIỆU GIÁ CỔ PHIẾU LỊCH SỬ (VNSTOCK)
# ======================================================
@st.cache_data(show_spinner=False, ttl=7200)
def load_price_data(ticker: str) -> pd.DataFrame:
    """
    Lấy dữ liệu giá cổ phiếu lịch sử qua Vnstock API.
    Ưu tiên tải từ cache CSV trước.
    """
    Vnstock = _get_vnstock()
    if Vnstock is None:
        return pd.DataFrame()
        
    ticker = ticker.upper()
    path = f"data/prices/{ticker}_vnstock.csv" # Đổi tên cache để tránh xung đột
    
    # Ngày bắt đầu và kết thúc (YYYY-MM-DD) - vnstock dùng định dạng này
    start_date_str = "2018-01-01"
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    
    # 🔹 Danh sách mã đã bị delisted (Ngày hủy niêm yết chính thức DD/MM/YYYY)
    delisted_info = {
        'FLC': '05/09/2023', 
        'GAB': '01/03/2024',
        'HAI': '01/08/2023',
    }
    ticker_upper = ticker.upper()

    # 1. Tải từ cache local
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            # Kiểm tra xem cache có cần cập nhật không
            if df.index.max().date() == datetime.now().date():
                return df
            # Nếu không, tiếp tục tải mới
            os.remove(path)
        except Exception:
            os.remove(path)

    # 2. Tải từ Vnstock với retry logic
    df = None
    max_retries = 2
    sources = ['VCI', 'TCBS']  # Thử nhiều nguồn
    
    for attempt in range(max_retries):
        for source in sources:
            try:
                stock = Vnstock().stock(symbol=ticker, source=source)
                df = stock.quote.history(start=start_date_str, end=end_date_str)

                if df is not None and not df.empty:
                    # Nếu có dữ liệu, thoát khỏi vòng lặp
                    break
            except Exception as e:
                logger.warning(f"Lần thử {attempt + 1}: Lỗi tải từ {source}: {str(e)[:100]}")
                continue
        
        if df is not None and not df.empty:
            break  # Đã có dữ liệu, thoát vòng ngoài
    
    if df is None or df.empty:
        st.warning(f"⚠️ Không tìm thấy dữ liệu cho {ticker} trên Vnstock sau {max_retries} lần thử với {len(sources)} nguồn.")
        return pd.DataFrame()

    # 🔹 Chuẩn hóa tên cột
    df.rename(columns={
        "time": "date", # Cột time thành date
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    }, inplace=True)
    
    # Thêm Adj Close (tạm thời bằng Close nếu không có sẵn)
    if 'adj_close' not in df.columns:
        df['adj_close'] = df['close']
        
    # Chọn các cột cần thiết và đảm bảo thứ tự
    required_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    df = df[[col for col in required_cols if col in df.columns]].copy()

    # Đảm bảo cột date là datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 🔹 SET DATE LÀM INDEX (Quan trọng cho biểu đồ)
    df = df.set_index('date')
    
    # 🔹 LÀM SẠCH DỮ LIỆU MẠNH MẼ: Loại bỏ giá = 0 hoặc flatline sau delisting
    df = df[df['close'] > 0].copy()
    df = df.dropna(subset=['close', 'open', 'high', 'low'])
    
    # 🔹 LỌC MÃ BỊ DELISTED: Chỉ giữ dữ liệu đến ngày delisting
    if ticker_upper in delisted_info:
        delisting_date_str = delisted_info[ticker_upper]
        # Chuyển đổi DD/MM/YYYY sang datetime
        delisting_date = datetime.strptime(delisting_date_str, "%d/%m/%Y")
        
        # Chỉ giữ lại các hàng có ngày <= ngày delisting (sử dụng index)
        df = df[df.index <= delisting_date].copy()
        logger.info(f"Lọc dữ liệu {ticker} đến ngày delisting: {delisting_date.date()}")
        
    # 🔹 KIỂM TRA TÍNH CHÍNH XÁC (Abnormal changes)
    if len(df) > 0:
        # Giá được trả về từ Vnstock thường đã được nhân 1000/10000 tùy nguồn, 
        # nhưng phần trăm thay đổi vẫn chính xác.
        price_change = df['close'].pct_change().abs()
        abnormal_days = price_change[price_change > 0.5]
        if len(abnormal_days) > 0:
            logger.warning(f"{ticker}: Phát hiện {len(abnormal_days)} ngày có biến động giá >50%")
    
    # Lưu vào cache local (giữ lại index date)
    os.makedirs("data/prices", exist_ok=True)
    df.to_csv(path, index=True)
    
    logger.info(f"✅ Đã tải {len(df)} ngày dữ liệu cho {ticker} từ Vnstock")
    return df


# ======================================================
# 💹 TẢI DỮ LIỆU GIÁ REAL-TIME (CÂU CHẤP)
# ======================================================
@st.cache_data(ttl=5, show_spinner=False) # Cache 5 giây để cập nhật
def load_realtime_price_quote(ticker: str) -> Optional[Dict]:
    """
    Lấy dữ liệu giá real-time (last trade quote) từ VNDirect API.
    Sử dụng cho hiển thị tiêu đề và số liệu chính trên Dashboard.
    (Giữ nguyên)
    """
    if not ticker:
        return None
        
    try:
        vnd_api = get_vndirect_api()
        # Hàm get_stock_price() đã bao gồm logic xử lý timeout
        return vnd_api.get_stock_price(ticker) 
    except Exception as e:
        logger.error(f"Lỗi tải giá real-time cho {ticker}: {e}")
        return None


# ======================================================
# 🔁 TẢI DỮ LIỆU KIỂM ĐỊNH GRANGER/TVAR THEO CẤU HÌNH SIDEBAR
# ======================================================
@st.cache_data(show_spinner=False, ttl=7200)
def load_granger_data(
    ticker: Optional[str] = None, data_type: str = "Content", time_period: str = "Before Scandal"
) -> pd.DataFrame:
    """
    Tải dữ liệu đã chuẩn hóa (thường là Log Return Price và Sentiment Score)
    cho các mô hình Kinh tế lượng (Granger, TVAR).
    (Giữ nguyên)
    """
    base_dir = "data"
    # Các thư mục này chứa dữ liệu đã được xử lý cho mô hình kinh tế lượng
    type_map = {"Content": "data", "Title": "data_title"} 
    period_map = {"Before Scandal": "before_scandals", "After Scandal": "after_scandals"}

    folder_name = f"{type_map.get(data_type, 'data')}_{period_map.get(time_period, 'before_scandals')}"
    data_dir = os.path.join(base_dir, folder_name)

    if not os.path.exists(data_dir):
        st.error(f"❌ Thư mục dữ liệu không tồn tại: `{data_dir}`")
        logger.error(f"Không tìm thấy thư mục: {data_dir}")
        return pd.DataFrame()

    # Nếu người dùng chọn mã cổ phiếu cụ thể
    if ticker:
        file_path = os.path.join(data_dir, f"{ticker.upper()}.xlsx")
        if os.path.exists(file_path):
            df = _safe_load_excel(file_path)
            return df
        else:
            st.warning(f"⚠️ Không tìm thấy file `{ticker}.xlsx` trong `{folder_name}/`.")
            logger.warning(f"Thiếu file: {file_path}")
            return pd.DataFrame()

    # Nếu không có ticker → hợp nhất toàn bộ file trong thư mục
    dfs = []
    for file in os.listdir(data_dir):
        if file.endswith((".xlsx", ".xls")) and not file.startswith("~$"):
            df = _safe_load_excel(os.path.join(data_dir, file))
            if not df.empty:
                df["ticker"] = file.replace(".xlsx", "").replace(".xls", "").upper()
                dfs.append(df)

    if not dfs:
        st.error(f"❌ Không tìm thấy file Excel nào trong `{folder_name}/`.")
        logger.error(f"Không có file Excel trong thư mục {data_dir}")
        return pd.DataFrame()

    st.info(f"📊 Đã hợp nhất dữ liệu Granger trong `{folder_name}/` ({len(dfs)} file).")
    return pd.concat(dfs, ignore_index=True)