import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import streamlit as st

@st.cache_data(show_spinner="Đang tính toán kiểm định Pearson...")
def pearson_test(df: pd.DataFrame, sentiment_col: str, variables: list) -> pd.DataFrame:
    """
    Thực hiện kiểm định tương quan Pearson giữa cột cảm xúc và các biến được chọn.

    Args:
        df (DataFrame): Dữ liệu chứa các cột cần kiểm định.
        sentiment_col (str): Tên cột biểu diễn điểm cảm xúc (ví dụ: 'label').
        variables (list): Danh sách các biến cần kiểm định với cảm xúc.

    Returns:
        DataFrame: Bảng kết quả gồm biến, hệ số tương quan, p-value và diễn giải.
    """
    results = []

    if sentiment_col not in df.columns:
        raise ValueError(f"⚠️ Cột '{sentiment_col}' không tồn tại trong DataFrame!")

    for v in variables:
        if v not in df.columns:
            # Bỏ qua biến không tồn tại
            continue

        valid_data = df[[sentiment_col, v]].dropna()

        if len(valid_data) < 3:
            # Không đủ dữ liệu
            continue

        # Kiểm tra chuỗi gần hằng (ví dụ: toàn 0 hoặc 1)
        if np.isclose(valid_data[sentiment_col].std(), 0) or np.isclose(valid_data[v].std(), 0):
            results.append({
                "Variable": v,
                "r_value": None,
                "p_value": None,
                "Relationship": "Chuỗi gần hằng - không thể kiểm định"
            })
            continue

        corr, pval = pearsonr(valid_data[sentiment_col], valid_data[v])

        if pval < 0.05:
            if corr > 0:
                relation = "Tuyến tính tích cực (p<0.05)"
            else:
                relation = "Tuyến tính tiêu cực (p<0.05)"
        else:
            relation = "Không có ý nghĩa thống kê (p≥0.05)"

        results.append({
            "Variable": v,
            "r_value": round(corr, 4),
            "p_value": round(pval, 4),
            "Relationship": relation
        })

    return pd.DataFrame(results)
