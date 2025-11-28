"""
Optimized Technical Indicators Module (Final Stable)
- Vectorized, NaN-safe, and performance-optimized
- Compatible with Streamlit visualization
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ==========================
# MOVING AVERAGES
# ==========================
def add_sma(data: pd.DataFrame, window: int, col: str = "Close", name: str = None) -> pd.DataFrame:
    try:
        name = name or f"SMA_{window}"
        data[name] = data[col].rolling(window, min_periods=1).mean().round(2)
        return data
    except Exception as e:
        logger.error(f"add_sma failed: {e}")
        return data


def add_ema(data: pd.DataFrame, span: int, col: str = "Close", name: str = None) -> pd.DataFrame:
    try:
        name = name or f"EMA_{span}"
        data[name] = data[col].ewm(span=span, adjust=False).mean().round(2)
        return data
    except Exception as e:
        logger.error(f"add_ema failed: {e}")
        return data


# ==========================
# OSCILLATORS
# ==========================
def add_rsi(data: pd.DataFrame, period: int = 14, col: str = "Close") -> pd.DataFrame:
    try:
        delta = data[col].diff().fillna(0)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Sử dụng ewm cho tính toán Average Gain/Loss
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        # Giữ nguyên logic RSI
        rs = avg_gain / (avg_loss + 1e-9)
        data["RSI"] = 100 - (100 / (1 + rs))
        data["RSI"] = data["RSI"].fillna(50).clip(0, 100).round(2)
        # Bổ sung các cột boolean cho phân tích
        data["RSI_Overbought"] = data["RSI"] > 70
        data["RSI_Oversold"] = data["RSI"] < 30
        return data
    except Exception as e:
        logger.error(f"add_rsi failed: {e}")
        return data
def add_macd(data: pd.DataFrame, col: str = "Close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    try:
        ema_fast = data[col].ewm(span=fast, adjust=False).mean()
        ema_slow = data[col].ewm(span=slow, adjust=False).mean()
        data["MACD"] = ema_fast - ema_slow
        data["MACD_Signal"] = data["MACD"].ewm(span=signal, adjust=False).mean()
        # 💡 Cải tiến: Đổi tên cột Histogram thành MACDH để nhất quán
        data["MACDH"] = data["MACD"] - data["MACD_Signal"]
        return data
    except Exception as e:
        logger.error(f"add_macd failed: {e}")
        return data


def add_stoch(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    try:
        high_roll = data["High"].rolling(k_period, min_periods=1).max()
        low_roll = data["Low"].rolling(k_period, min_periods=1).min()
        data["Stoch_K"] = 100 * (data["Close"] - low_roll) / (high_roll - low_roll + 1e-9)
        data["Stoch_D"] = data["Stoch_K"].rolling(d_period, min_periods=1).mean()
        data["Stoch_K"] = data["Stoch_K"].clip(0, 100).round(2)
        data["Stoch_D"] = data["Stoch_D"].clip(0, 100).round(2)
        return data
    except Exception as e:
        logger.error(f"add_stoch failed: {e}")
        return data


# ==========================
# TREND & VOLATILITY
# ==========================
def add_adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    try:
        high, low, close = data["High"], data["Low"], data["Close"]
        # True Range (TR)
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        
        # Directional Movement (DM)
        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        # Lọc DM+ và DM-
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        # Average True Range (ATR)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        
        # Directional Index (DI)
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9))
        
        # Average Directional Index (ADX)
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        
        # 💡 Cải tiến: Chuẩn hóa tên cột DI
        data["ADX"] = dx.ewm(alpha=1 / period, adjust=False).mean().round(2)
        data["ADX_+DI"] = plus_di.round(2)
        data["ADX_-DI"] = minus_di.round(2)
        return data
    except Exception as e:
        logger.error(f"add_adx failed: {e}")
        return data


def add_bollinger_bands(data: pd.DataFrame, col: str = "Close", window: int = 20, std_dev: float = 2) -> pd.DataFrame:
    try:
        ma = data[col].rolling(window, min_periods=1).mean()
        std = data[col].rolling(window, min_periods=1).std()
        data["BB_Middle"] = ma
        data["BB_Upper"] = ma + std_dev * std
        data["BB_Lower"] = ma - std_dev * std
        # Tính toán Bollinger Band Width (BBW)
        data["BB_Width"] = ((data["BB_Upper"] - data["BB_Lower"]) / ma.replace(0, np.nan)).fillna(0).round(3)
        return data
    except Exception as e:
        logger.error(f"add_bollinger_bands failed: {e}")
        return data


def add_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    try:
        high, low, close = data["High"], data["Low"], data["Close"]
        # True Range (TR)
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        data["ATR"] = tr.ewm(alpha=1 / period, adjust=False).mean().round(2)
        return data
    except Exception as e:
        logger.error(f"add_atr failed: {e}")
        return data


# ==========================
# VOLUME
# ==========================
def add_obv(data: pd.DataFrame) -> pd.DataFrame:
    try:
        direction = np.sign(data["Close"].diff()).fillna(0)
        obv = (direction * data["Volume"]).cumsum()
        data["OBV"] = obv
        data["OBV_MA"] = obv.rolling(20, min_periods=1).mean().round(2)
        return data
    except Exception as e:
        logger.error(f"add_obv failed: {e}")
        return data


def add_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """Volume Weighted Average Price"""
    try:
        tp = (data["High"] + data["Low"] + data["Close"]) / 3
        # VWAP là tích lũy (tp * volume) chia cho tích lũy volume
        data["VWAP"] = (tp * data["Volume"]).cumsum() / (data["Volume"].cumsum() + 1e-9)
        data["VWAP"] = data["VWAP"].round(2)
        return data
    except Exception as e:
        logger.error(f"add_vwap failed: {e}")
        return data


# ==========================
# FIBONACCI
# ==========================
def add_fibonacci_levels(data: pd.DataFrame, lookback_period: int = 50) -> Dict[str, float]:
    """Return Fibonacci retracement levels without mutating dataframe"""
    try:
        if len(data) < lookback_period:
            return {}
        recent = data.tail(lookback_period)
        high, low = recent["High"].max(), recent["Low"].min()
        diff = high - low
        return {
            "Fib_0%": round(high, 2),
            "Fib_23.6%": round(high - 0.236 * diff, 2),
            "Fib_38.2%": round(high - 0.382 * diff, 2),
            "Fib_50%": round(high - 0.5 * diff, 2),
            "Fib_61.8%": round(high - 0.618 * diff, 2),
            "Fib_78.6%": round(high - 0.786 * diff, 2),
            "Fib_100%": round(low, 2),
        }
    except Exception as e:
        logger.error(f"add_fibonacci_levels failed: {e}")
        return {}


# ==========================
# INDICATOR SUMMARY
# ==========================
def get_indicator_summary(data: pd.DataFrame) -> Dict[str, dict]:
    if data.empty:
        return {}
    summary = {}
    latest = data.iloc[-1]

    try:
        # RSI
        if "RSI" in data.columns:
            val = float(latest["RSI"])
            signal = "BUY (Oversold)" if val < 30 else "SELL (Overbought)" if val > 70 else "HOLD (Neutral)"
            summary["RSI"] = {
                "Giá trị": round(val, 1),
                "Tín hiệu": signal,
            }

        # MACD
        if {"MACD", "MACD_Signal"}.issubset(data.columns):
            macd, sig = latest["MACD"], latest["MACD_Signal"]
            # Tín hiệu được xác định bởi sự giao cắt
            prev_macd = data["MACD"].iloc[-2] if len(data) >= 2 else 0
            prev_sig = data["MACD_Signal"].iloc[-2] if len(data) >= 2 else 0
            
            if macd > sig and prev_macd <= prev_sig:
                signal = "BUY (Cắt lên)"
            elif macd < sig and prev_macd >= prev_sig:
                signal = "SELL (Cắt xuống)"
            elif macd > sig:
                signal = "Bullish"
            else:
                signal = "Bearish"
                
            summary["MACD"] = {
                "MACD": round(macd, 2),
                "Signal Line": round(sig, 2),
                "Tín hiệu": signal,
            }

        # Bollinger Bands
        if {"BB_Upper", "BB_Lower", "Close"}.issubset(data.columns):
            close, up, low = latest["Close"], latest["BB_Upper"], latest["BB_Lower"]
            if close >= up:
                pos = "Near Upper Band (Quá mua)"
                signal = "SELL"
            elif close <= low:
                pos = "Near Lower Band (Quá bán)"
                signal = "BUY"
            else:
                pos = "Within Bands"
                signal = "HOLD"
                
            summary["Bollinger_Bands"] = {
                "Vị trí": pos,
                "Độ rộng": round(latest.get("BB_Width", 0), 3),
                "Tín hiệu": signal,
            }
        
        # ADX (Directional Movement Index)
        if {"ADX", "ADX_+DI", "ADX_-DI"}.issubset(data.columns):
            adx, pdi, mdi = latest["ADX"], latest["ADX_+DI"], latest["ADX_-DI"]
            if adx < 20:
                trend = "Weak/Non-trending"
            elif adx > 25:
                trend = "Strong Trend"
            else:
                trend = "Moderate Trend"
            
            if pdi > mdi:
                direction = "Bullish"
            elif mdi > pdi:
                direction = "Bearish"
            else:
                direction = "Neutral"

            summary["ADX"] = {
                "Giá trị ADX": round(adx, 2),
                "Độ mạnh Trend": trend,
                "Tín hiệu DI": f"{direction} (+DI={pdi:.2f}, -DI={mdi:.2f})",
            }


        return summary
    except Exception as e:
        logger.error(f"get_indicator_summary failed: {e}")
        return summary