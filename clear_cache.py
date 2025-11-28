"""
Script để clear Streamlit cache khi cần thiết
Chạy: python clear_cache.py
"""

import streamlit as st
import shutil
import os

def clear_streamlit_cache():
    """Clear tất cả cache của Streamlit"""
    cache_dirs = [
        '.streamlit/cache',
        '__pycache__',
        'config/__pycache__',
        'models/__pycache__',
        'ui/__pycache__',
        'utils/__pycache__',
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Đã xóa: {cache_dir}")
            except Exception as e:
                print(f"⚠️  Không thể xóa {cache_dir}: {e}")
        else:
            print(f"ℹ️  Không tồn tại: {cache_dir}")
    
    print("\n✨ Hoàn tất! Khởi động lại app để áp dụng.")

if __name__ == "__main__":
    clear_streamlit_cache()
