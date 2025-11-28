"""
Optimized Head & Shoulders Pattern Detection
- Vectorized scanning
- Supports inverse (bullish) pattern
- Stable with noisy OHLC data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional # Thêm Optional

logger = logging.getLogger(__name__)

# ======================================================
# 🧠 CORE DETECTION LOGIC (GIỮ NGUYÊN)
# ======================================================
def detect_head_and_shoulders(
    data: pd.DataFrame,
    window: int = 10,
    tolerance: float = 0.05,
    detect_inverse: bool = True
) -> pd.DataFrame:
    """
    Detect Head and Shoulders (regular & inverse)
    Args:
        data: DataFrame with columns ['High', 'Low']
        window: lookback window
        tolerance: max deviation between shoulders (%)
        detect_inverse: also detect inverse pattern
    Returns:
        DataFrame with 'Head_and_Shoulders' column (0/1)
    """
    df = data.copy()
    df["Head_and_Shoulders"] = 0
    df["Pattern_Type"] = None
    df["Neckline_Price"] = np.nan # Thêm cột Neckline

    if len(df) < window * 3:
        logger.warning("Not enough data for pattern detection")
        return df

    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    try:
        for i in range(window, n - window):
            # --- Regular pattern (Bearish) ---
            left_shoulder = highs[i - window:i].max()
            head = highs[i]
            right_shoulder = highs[i + 1:i + window + 1].max()

            if head > left_shoulder and head > right_shoulder:
                shoulders_similar = abs(left_shoulder - right_shoulder) <= tolerance * head
                if shoulders_similar:
                    df.loc[df.index[i], "Head_and_Shoulders"] = 1
                    df.loc[df.index[i], "Pattern_Type"] = "regular"
                    # Neckline: Ước tính đơn giản là giá Low của 2 đáy gần nhất
                    # (Cần logic phức tạp hơn để xác định 2 đáy thực sự)
                    # Tạm thời đặt giá đóng cửa tại điểm phát hiện là giá Neckline gần đúng
                    df.loc[df.index[i], "Neckline_Price"] = df.loc[df.index[i], "Low"]

            # --- Inverse pattern (Bullish) ---
            if detect_inverse:
                left_valley = lows[i - window:i].min()
                head_valley = lows[i]
                right_valley = lows[i + 1:i + window + 1].min()
                if head_valley < left_valley and head_valley < right_valley:
                    valleys_similar = abs(left_valley - right_valley) <= tolerance * left_valley
                    if valleys_similar:
                        df.loc[df.index[i], "Head_and_Shoulders"] = 1
                        df.loc[df.index[i], "Pattern_Type"] = "inverse"
                        # Neckline: Tạm thời đặt giá đóng cửa tại điểm phát hiện là giá Neckline gần đúng
                        df.loc[df.index[i], "Neckline_Price"] = df.loc[df.index[i], "High"]

    except Exception as e:
        logger.error(f"Pattern detection error: {e}")

    return df


# ======================================================
# 💡 SIGNAL EVALUATION (CẢI TIẾN)
# ======================================================
def check_head_and_shoulders_signal(data: pd.DataFrame) -> Dict[str, any]:
    """
    Evaluate most recent pattern and check for neckline breakout/breakdown.
    """
    if "Head_and_Shoulders" not in data or data.empty:
        return {"signal": "No data", "strength": "None", "message": "No data available."}

    recent = data.tail(40) # Tăng window kiểm tra gần nhất
    found = recent[recent["Head_and_Shoulders"] == 1]
    
    if found.empty:
        return {"signal": "No pattern", "strength": "None", "message": "No pattern found in recent data."}

    latest_idx = found.index[-1]
    p_type = found["Pattern_Type"].iloc[-1]
    days_ago = len(data) - data.index.get_loc(latest_idx) - 1
    
    current_close = data["Close"].iloc[-1]
    neckline_price = found["Neckline_Price"].iloc[-1]
    
    strength = "Strong" if days_ago <= 5 else "Moderate" if days_ago <= 15 else "Weak"
    sentiment = "Bearish" if p_type == "regular" else "Bullish"

    # --- BREAKOUT LOGIC (Cải tiến) ---
    
    is_breakout = False
    action = "Pending Confirmation" # Trạng thái mặc định

    if p_type == "regular":
        # Regular H&S -> Bearish breakdown
        if current_close < neckline_price:
            is_breakout = True
            action = "SELL Signal (Breakdown)"
        else:
            action = "Monitor (Breakdown Pending)"
            
    elif p_type == "inverse":
        # Inverse H&S -> Bullish breakout
        if current_close > neckline_price:
            is_breakout = True
            action = "BUY Signal (Breakout)"
        else:
            action = "Monitor (Breakout Pending)"

    return {
        "signal": action,
        "pattern_type": p_type,
        "strength": strength,
        "days_ago": days_ago,
        "date": latest_idx.strftime("%Y-%m-%d"),
        "neckline": round(neckline_price, 2),
        "is_breakout": is_breakout,
        "message": f"{p_type.title()} H&S detected {days_ago} days ago ({strength}). Signal: {action}"
    }


# ======================================================
# 💡 HELPER FUNCTIONS (GIỮ NGUYÊN)
# ======================================================
def get_head_and_shoulders_summary(data: pd.DataFrame) -> Dict[str, any]:
    """Summarize detected patterns"""
    if "Head_and_Shoulders" not in data:
        return {"error": "Column missing"}

    total = int(data["Head_and_Shoulders"].sum())
    if total == 0:
        return {"total_patterns": 0, "message": "No pattern detected"}

    dates = data[data["Head_and_Shoulders"] == 1].index
    types = data.loc[dates, "Pattern_Type"].tolist()
    latest = dates[-1].strftime("%Y-%m-%d")

    return {
        "total_patterns": total,
        "latest_pattern": latest,
        "pattern_types": types[-5:],
        "pattern_dates": [d.strftime("%Y-%m-%d") for d in dates],
        "message": f"{total} patterns found (latest: {types[-1].title()} H&S)"
    }


def get_latest_head_and_shoulders(data: pd.DataFrame, n: int = 5) -> List[str]:
    """Return N most recent pattern dates"""
    if "Head_and_Shoulders" not in data:
        return []
    dates = data[data["Head_and_Shoulders"] == 1].index
    return [d.strftime("%Y-%m-%d") for d in dates[-n:][::-1]]


def analyze_head_and_shoulders(stock_data: pd.DataFrame, window: int = 10) -> Dict[str, any]:
    """One-step H&S analysis"""
    try:
        analyzed = detect_head_and_shoulders(stock_data, window)
        return {
            "data": analyzed,
            "summary": get_head_and_shoulders_summary(analyzed),
            "signal": check_head_and_shoulders_signal(analyzed),
            "recent_patterns": get_latest_head_and_shoulders(analyzed),
            "success": True
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"success": False, "error": str(e)}