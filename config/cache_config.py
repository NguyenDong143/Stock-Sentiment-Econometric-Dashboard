# ======================================================
# 🚀 Cache Configuration for Performance Optimization
# ======================================================

"""
Cấu hình tối ưu hóa cache cho toàn bộ ứng dụng
Giúp giảm thời gian load và tăng trải nghiệm người dùng
"""

# ======================================================
# CACHE TTL (Time To Live) Settings
# ======================================================
# Thời gian cache (giây)

# Data loaders
DATA_LOADER_TTL = 300  # 5 phút - dữ liệu sentiment và giá
PRICE_DATA_TTL = 3600  # 1 giờ - dữ liệu giá lịch sử
REALTIME_PRICE_TTL = 5  # 5 giây - giá realtime

# API calls
API_COMPANY_INFO_TTL = 3600  # 1 giờ - thông tin công ty
API_STOCK_QUOTE_TTL = 5  # 5 giây - quote realtime

# Model computations
GRANGER_TEST_TTL = 600  # 10 phút - kết quả Granger test
PEARSON_TEST_TTL = 600  # 10 phút - kết quả Pearson test
TVAR_MODEL_TTL = 600  # 10 phút - mô hình TVAR

# Resources (models không bao giờ clear)
PHOBERT_MODEL_TTL = None  # Vĩnh viễn - PhoBERT model

# ======================================================
# CACHE SIZE Settings
# ======================================================
# Giới hạn kích thước cache (MB)
MAX_CACHE_SIZE_MB = 500

# ======================================================
# OPTIMIZATION FLAGS
# ======================================================
# Bật/tắt các tính năng tối ưu

# Lazy loading
ENABLE_LAZY_LOADING = True

# Debouncing (milliseconds)
DEBOUNCE_DELAY = 300

# Show spinner
SHOW_DATA_SPINNER = True
SHOW_MODEL_SPINNER = True
SHOW_API_SPINNER = False

# ======================================================
# DEBUG Settings
# ======================================================
# Hiển thị thông tin cache (chỉ dùng khi debug)
SHOW_CACHE_INFO = False
LOG_CACHE_HITS = False

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def get_cache_config(component: str) -> dict:
    """
    Lấy cấu hình cache cho một component cụ thể
    
    Args:
        component: Tên component ('data_loader', 'model', 'api', etc.)
    
    Returns:
        dict: Cấu hình cache
    """
    configs = {
        'data_loader': {
            'ttl': DATA_LOADER_TTL,
            'show_spinner': SHOW_DATA_SPINNER
        },
        'price_data': {
            'ttl': PRICE_DATA_TTL,
            'show_spinner': SHOW_DATA_SPINNER
        },
        'realtime_price': {
            'ttl': REALTIME_PRICE_TTL,
            'show_spinner': SHOW_API_SPINNER
        },
        'granger_test': {
            'ttl': GRANGER_TEST_TTL,
            'show_spinner': SHOW_MODEL_SPINNER
        },
        'pearson_test': {
            'ttl': PEARSON_TEST_TTL,
            'show_spinner': SHOW_MODEL_SPINNER
        },
        'tvar_model': {
            'ttl': TVAR_MODEL_TTL,
            'show_spinner': SHOW_MODEL_SPINNER
        },
        'phobert_model': {
            'ttl': PHOBERT_MODEL_TTL,
            'show_spinner': False
        },
        'api_company_info': {
            'ttl': API_COMPANY_INFO_TTL,
            'show_spinner': SHOW_API_SPINNER
        }
    }
    
    return configs.get(component, {'ttl': 300, 'show_spinner': True})


def clear_all_cache():
    """Xóa toàn bộ cache (dùng khi cần refresh dữ liệu)"""
    try:
        import streamlit as st
        st.cache_data.clear()
        st.cache_resource.clear()
        return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False


def get_cache_stats():
    """Lấy thống kê cache (chỉ dùng khi debug)"""
    if not SHOW_CACHE_INFO:
        return None
    
    # TODO: Implement cache statistics tracking
    return {
        'enabled': True,
        'max_size_mb': MAX_CACHE_SIZE_MB,
        'components_cached': [
            'data_loader', 'price_data', 'granger_test',
            'pearson_test', 'tvar_model', 'phobert_model'
        ]
    }
