"""
Optimized Chart Module for Streamlit Stock Dashboard (Complete Fixed Version)
- FIXED: Nến không bị dính nhau, có khoảng cách tự nhiên
- FIXED: Trục X hiển thị ngày tháng rõ ràng
- FIXED: Loại bỏ khoảng trống cuối tuần thông minh
- Tối ưu hóa màu sắc MA để nổi bật hơn
- Nền biểu đồ hòa quyện (plot_bgcolor == paper_bgcolor)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ==========================================================
# 📊 CONFIGURATION CONSTANTS
# ==========================================================
DEFAULT_HEIGHT = 700
DEFAULT_VISIBLE_DAYS = 60
MIN_DATA_POINTS = 2
MAX_RENDER_POINTS = 2000

# Subplot heights
PRICE_HEIGHT_1_INDICATOR = 0.70
PRICE_HEIGHT_2_INDICATORS = 0.65
PRICE_HEIGHT_3PLUS_INDICATORS = 0.55
INDICATOR_HEIGHT = 0.12

# RSI levels
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL = 50

# Stochastic levels
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20
STOCH_NEUTRAL = 50

# ADX levels
ADX_STRONG_TREND = 25
ADX_WEAK_TREND = 20

# ==========================================================
# 🎨 COLOR SCHEME (Tối ưu cho Dark Mode và tương phản)
# ==========================================================
COLORS = {
    "bullish": "#00E676", # Bright Green (Tăng) - màu xanh lá sáng hơn
    "bullish_line": "#00C853", # Viền nến tăng
    "bearish": "#FF5252", # Bright Red (Giảm) - màu đỏ sáng hơn
    "bearish_line": "#D32F2F", # Viền nến giảm
    "neutral": "#9CA3AF", # Gray (Trung tính)
    "volume": "#9333EA", # Violet
    # Tweak: Màu sáng hơn, nổi bật hơn cho MA
    "ma_short": "#38BDF8", # Sky Blue (SMA 20)
    "ma_long": "#F59E0B", # Amber (SMA 50)
    "ema_short": "#F06292", # Pink (EMA 12)
    "ema_long": "#9575CD", # Purple (EMA 26)
    "rsi": "#EC4899", # Pink
    "macd": "#34D399", # Teal
    "bb_upper": "#F87171",
    "bb_lower": "#34D399",
    "bb_middle": "#64748B", # Slate đậm hơn cho đường giữa
    "support": "#10B981",
    "resistance": "#F87171",
    "pattern": "#FBBF24", # Yellow
    "stoch": "#F97316", # Orange
    "adx_di": "#38BDF8", # Sky Blue
    "adx": "#A78BFA", # Violet
}


# ==========================================================
# 🛡️ VALIDATION & HELPER FUNCTIONS
# ==========================================================
def validate_data(data: pd.DataFrame, chart_type: str) -> bool:
    """Validate dữ liệu đầy đủ trước khi vẽ biểu đồ.
    
    Args:
        data: DataFrame cần validate
        chart_type: "Candle" hoặc "Line"
        
    Raises:
        ValueError: Nếu dữ liệu không hợp lệ
        
    Returns:
        True nếu dữ liệu hợp lệ
    """
    if data is None or data.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Kiểm tra columns bắt buộc
    required_cols = ['Open', 'High', 'Low', 'Close'] if chart_type == "Candle" else ['Close']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Kiểm tra index phải là DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    
    # Kiểm tra số lượng điểm dữ liệu tối thiểu
    if len(data) < MIN_DATA_POINTS:
        raise ValueError(f"Need at least {MIN_DATA_POINTS} data points, got {len(data)}")
    
    return True


def clean_indicator_data(data: pd.DataFrame, col_name: str) -> pd.Series:
    """Loại bỏ NaN/inf trong indicator trước khi plot.
    
    Args:
        data: DataFrame chứa indicator
        col_name: Tên cột indicator
        
    Returns:
        Series đã được làm sạch
    """
    if col_name not in data.columns:
        return pd.Series(dtype=float)
    
    clean_data = data[col_name].replace([np.inf, -np.inf], np.nan)
    return clean_data.dropna()


def optimize_data_for_rendering(data: pd.DataFrame, max_points: int = MAX_RENDER_POINTS) -> pd.DataFrame:
    """Giảm số điểm vẽ nếu quá nhiều để tối ưu performance.
    
    Args:
        data: DataFrame cần tối ưu
        max_points: Số điểm tối đa
        
    Returns:
        DataFrame đã được tối ưu
    """
    if len(data) <= max_points:
        return data
    
    # Downsample thông minh: giữ lại điểm quan trọng
    step = len(data) // max_points
    return data.iloc[::step]


def create_advanced_chart(
    data: pd.DataFrame,
    chart_type: str = "Candle",
    indicators: List[str] = None,
    levels: Dict = None,
    patterns: pd.DataFrame = None,
    title: str = "Stock Chart",
    height: int = DEFAULT_HEIGHT,
    show_volume: bool = True,
    default_visible_days: int = DEFAULT_VISIBLE_DAYS,
) -> go.Figure:
    """Tạo biểu đồ kỹ thuật chuyên nghiệp cho chứng khoán.
    
    Args:
        data: DataFrame với index DatetimeIndex, columns gồm OHLCV và các indicators
        chart_type: "Candle" hoặc "Line"
        indicators: List tên indicators cần hiển thị, ví dụ:
            - "SMA_20", "SMA_50", "EMA_12", "EMA_26"
            - "Bollinger_Bands" (cần BB_Upper, BB_Lower, BB_Middle)
            - "RSI", "MACD", "Stochastic", "ADX", "VWAP"
        levels: Dict {tên: giá trị} cho support/resistance, ví dụ:
            {"Support 1": 50000, "Resistance 1": 55000}
        patterns: DataFrame với column "Head_and_Shoulders" (1/0)
        title: Tiêu đề biểu đồ
        height: Chiều cao pixel (default: 700)
        show_volume: Hiển thị volume chart
        default_visible_days: Số ngày hiển thị ban đầu (default: 60)
    
    Returns:
        go.Figure: Plotly figure object
        
    Raises:
        ValueError: Nếu data không hợp lệ
        
    Example:
        >>> fig = create_advanced_chart(
        ...     data=df,
        ...     indicators=["SMA_20", "RSI", "MACD"],
        ...     title="HOSE:VNM"
        ... )
    """
    # Validation đầu vào
    validate_data(data, chart_type)
    
    # Loại bỏ các hàng có giá trị NaN hoặc 0 trong OHLC để tránh nến bị méo
    data = data.copy()
    if chart_type == "Candle":
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        data = data[(data['Open'] > 0) & (data['High'] > 0) & (data['Low'] > 0) & (data['Close'] > 0)]

    indicators = indicators or []
    levels = levels or {}
    patterns = patterns or pd.DataFrame()

    # --- 1. Subplot structure ---
    subplot_titles, row_heights = ["Price"], [0.60]  # Tăng tỉ lệ Price chart
    
    indicator_plots = []
    if show_volume: indicator_plots.append(("Volume", 0.12))  # Giảm Volume
    if "RSI" in indicators: indicator_plots.append(("RSI", 0.12))
    if "MACD" in indicators: indicator_plots.append(("MACD", 0.12))
    if "Stochastic" in indicators: indicator_plots.append(("Stochastic", 0.12))
    if "ADX" in indicators: indicator_plots.append(("ADX", 0.12))

    # Điều chỉnh base_height dựa trên số indicator
    base_height = 0.60
    if len(indicator_plots) == 1:
        base_height = PRICE_HEIGHT_1_INDICATOR
    elif len(indicator_plots) == 2:
        base_height = PRICE_HEIGHT_2_INDICATORS
    elif len(indicator_plots) >= 3:
        base_height = PRICE_HEIGHT_3PLUS_INDICATORS
    
    row_heights = [base_height] + [h for _, h in indicator_plots]
    subplot_titles.extend([name for name, _ in indicator_plots])

    total_rows = len(subplot_titles)
    
    # Tạo specs cho secondary_y (cần cho OBV)
    specs = [[{"secondary_y": True}] for _ in range(total_rows)]
    
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        specs=specs,
    )

    # --- 2. Main chart (Row 1) ---
    row = 1
    _add_main_chart(fig, data, chart_type, row)
    _add_price_indicators(fig, data, indicators, row)
    _add_levels(fig, data, levels, row)
    _add_patterns(fig, data, patterns, row)

    # --- 3. Extra subplots (Row 2+) ---
    current_row = 1
    if show_volume: 
        current_row += 1
        _add_volume_chart(fig, data, current_row)
    if "RSI" in indicators: 
        current_row += 1
        _add_rsi_chart(fig, data, current_row)
    if "MACD" in indicators: 
        current_row += 1
        _add_macd_chart(fig, data, current_row)
    if "Stochastic" in indicators: 
        current_row += 1
        _add_stoch_chart(fig, data, current_row)
    if "ADX" in indicators: 
        current_row += 1
        _add_adx_chart(fig, data, current_row)

    # --- 4. Final Layout Update ---
    _update_dark_layout(fig, title, height, total_rows, data, default_visible_days)
    return fig


# ==========================================================
# COMPONENTS
# ==========================================================
def _add_main_chart(fig, data, chart_type, row):
    try:
        if chart_type == "Candle":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name="Price",
                    # Màu sắc và viền cho nến tăng
                    increasing_fillcolor=COLORS["bullish"],
                    increasing_line_color=COLORS["bullish_line"],
                    increasing_line_width=1.5,
                    # Màu sắc và viền cho nến giảm
                    decreasing_fillcolor=COLORS["bearish"],
                    decreasing_line_color=COLORS["bearish_line"],
                    decreasing_line_width=1.5,
                    # Hover template đẹp hơn
                    hovertext=[
                        f"<b>Ngày:</b> {idx.strftime('%Y-%m-%d')}<br>"
                        f"<b>Mở:</b> {row['Open']:,.0f}<br>"
                        f"<b>Cao:</b> {row['High']:,.0f}<br>"
                        f"<b>Thấp:</b> {row['Low']:,.0f}<br>"
                        f"<b>Đóng:</b> {row['Close']:,.0f}<br>"
                        f"<b>Thay đổi:</b> {((row['Close'] - row['Open']) / row['Open'] * 100):+.2f}%"
                        for idx, row in data.iterrows()
                    ],
                    hoverinfo="text",
                ),
                row=row, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name="Close",
                    line=dict(color=COLORS["neutral"], width=2),
                    hovertemplate="<b>Close</b><br>" +
                                  "Ngày: %{x|%d/%m/%Y}<br>" +
                                  "Giá: %{y:,.2f}<br>" +
                                  "<extra></extra>",
                ),
                row=row, col=1,
            )
    except Exception as e:
        logger.warning(f"Main chart error: {e}")


def _add_price_indicators(fig, data, indicators, row):
    try:
        # MA/EMA (Sử dụng màu sắc và độ dày đã tweak)
        ma_map = {
            "SMA_20": (COLORS["ma_short"], 2.2), 
            "SMA_50": (COLORS["ma_long"], 2.2),
            "EMA_12": (COLORS["ema_short"], 1.8), 
            "EMA_26": (COLORS["ema_long"], 1.8)
        }
        # Ensure indicators align with valid price points to avoid "tails"
        valid_mask = data['Close'].notna() & (data['Close'] > 0) if 'Close' in data.columns else pd.Series(False, index=data.index)
        for name, (color, width) in ma_map.items():
            if name in indicators and name in data.columns:
                ma_data = data[name].copy()
                # Align MA/EMA to valid price points and drop NaNs
                ma_data = ma_data[valid_mask].dropna()
                if ma_data.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=ma_data.index,
                        y=ma_data,
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=width),
                        connectgaps=False,
                        hovertemplate=f"<b>{name}</b><br>" +
                                      "Ngày: %{x|%d/%m/%Y}<br>" +
                                      "Giá trị: %{y:,.2f}<br>" +
                                      "<extra></extra>",
                    ),
                    row=row, col=1,
                )

        # Bollinger Bands
        if "Bollinger_Bands" in indicators and all(c in data.columns for c in ["BB_Upper", "BB_Lower", "BB_Middle"]):
            # FIXED: Lọc BB để đồng bộ với dữ liệu giá
            valid_mask = data['Close'].notna() & (data['Close'] > 0)
            bb_upper = data["BB_Upper"][valid_mask].dropna()
            bb_lower = data["BB_Lower"][valid_mask].dropna()
            bb_middle = data["BB_Middle"][valid_mask].dropna()
            
            if not bb_upper.empty:
                fig.add_trace(go.Scatter(x=bb_upper.index, y=bb_upper, mode="lines",
                                         line=dict(color=COLORS["bb_upper"], dash="dot", width=1.5), 
                                         name="BB Upper", connectgaps=False,
                                         hovertemplate="<b>BB Upper</b><br>Ngày: %{x|%d/%m/%Y}<br>Giá trị: %{y:,.2f}<extra></extra>"), row=row, col=1)
            if not bb_lower.empty:
                fig.add_trace(go.Scatter(x=bb_lower.index, y=bb_lower, mode="lines",
                                         fill="tonexty", fillcolor="rgba(52,211,153,0.08)",
                                         line=dict(color=COLORS["bb_lower"], dash="dot", width=1.5), 
                                         name="BB Lower", connectgaps=False,
                                         hovertemplate="<b>BB Lower</b><br>Ngày: %{x|%d/%m/%Y}<br>Giá trị: %{y:,.2f}<extra></extra>"), row=row, col=1)
            # BB Middle (Tweak: Màu đậm hơn, độ dày 2.5)
            if not bb_middle.empty:
                fig.add_trace(go.Scatter(x=bb_middle.index, y=bb_middle, mode="lines",
                                         line=dict(color=COLORS["bb_middle"], width=2.5, dash="dashdot"), 
                                         name="BB Mid", connectgaps=False,
                                         hovertemplate="<b>BB Mid</b><br>Ngày: %{x|%d/%m/%Y}<br>Giá trị: %{y:,.2f}<extra></extra>"), row=row, col=1)
        
        # VWAP
        if "VWAP" in indicators and "VWAP" in data.columns:
            # FIXED: Lọc VWAP để đồng bộ với dữ liệu giá
            valid_mask = data['Close'].notna() & (data['Close'] > 0)
            vwap_data = data["VWAP"][valid_mask].dropna()
            if not vwap_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=vwap_data.index,
                        y=vwap_data,
                        mode="lines",
                        name="VWAP",
                        line=dict(color=COLORS["volume"], width=2.0, dash="dash"),
                        connectgaps=False,  # Không nối khoảng trống
                        hovertemplate="<b>VWAP</b><br>Ngày: %{x|%d/%m/%Y}<br>Giá trị: %{y:,.2f}<extra></extra>",
                    ),
                    row=row, col=1,
                )
    except Exception as e:
        logger.warning(f"Price Indicator error: {e}")


def _add_volume_chart(fig, data, row):
    try:
        if "Volume" in data.columns:
            if "Close" in data.columns and len(data) > 1:
                colors = np.where(data["Close"].diff().fillna(0) >= 0, COLORS["bullish"], COLORS["bearish"])
            else:
                colors = COLORS["volume"] 
                
            fig.add_trace(
                go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color=colors, opacity=0.8),
                row=row, col=1,
            )
            
            # Thêm OBV vào trục Y phụ bằng secondary_y
            if "OBV" in data.columns:
                obv_data = clean_indicator_data(data, "OBV")
                if not obv_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=obv_data.index,
                            y=obv_data,
                            mode="lines",
                            name="OBV",
                            line=dict(color=COLORS["volume"], width=1.5),
                            hovertemplate="<b>OBV</b><br>" +
                                          "Ngày: %{x|%d/%m/%Y}<br>" +
                                          "Giá trị: %{y:,.0f}<br>" +
                                          "<extra></extra>",
                        ),
                        row=row, col=1,
                        secondary_y=True,
                    )
                    # Cấu hình trục Y phụ cho OBV
                    fig.update_yaxes(
                        title_text="OBV",
                        showgrid=False,
                        titlefont=dict(color=COLORS["volume"], size=10),
                        tickfont=dict(color=COLORS["volume"], size=10),
                        row=row, col=1,
                        secondary_y=True,
                    )
    except Exception as e:
        logger.warning(f"Volume error: {e}")


def _add_rsi_chart(fig, data, row):
    if "RSI" not in data.columns:
        return
    rsi_data = clean_indicator_data(data, "RSI")
    if rsi_data.empty:
        return
    fig.add_trace(go.Scatter(
        x=rsi_data.index, 
        y=rsi_data.values, 
        mode="lines", 
        name="RSI",
        line=dict(color=COLORS["rsi"], width=2.0),
        connectgaps=False,
        hovertemplate="<b>RSI</b><br>" +
                      "Ngày: %{x|%d/%m/%Y}<br>" +
                      "Giá trị: %{y:.2f}<br>" +
                      "<extra></extra>",
    ), row=row, col=1)
    for y, c in [(RSI_OVERBOUGHT, COLORS["bearish"]), (RSI_NEUTRAL, COLORS["neutral"]), (RSI_OVERSOLD, COLORS["bullish"])]:
        fig.add_hline(y=y, line=dict(color=c, dash="dash", width=1.0), row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=row, col=1)


def _add_macd_chart(fig, data, row):
    if "MACD" not in data.columns:
        return
    try:
        # Use cleaned series and align with valid price points to avoid tails
        hist_col = "MACDH" if "MACDH" in data.columns else "MACD_Histogram"
        hist = clean_indicator_data(data, hist_col)
        macd = clean_indicator_data(data, "MACD")
        signal = clean_indicator_data(data, "MACD_Signal")

        if not hist.empty:
            colors = np.where(hist >= 0, COLORS["bullish"], COLORS["bearish"])
            fig.add_trace(go.Bar(x=hist.index, y=hist.values,
                                 name="MACDH",
                                 marker_color=colors, opacity=0.8),
                          row=row, col=1)
        if not macd.empty:
            fig.add_trace(go.Scatter(x=macd.index, y=macd.values, mode="lines", name="MACD",
                                     line=dict(color=COLORS["macd"], width=2.0), connectgaps=False,
                                     hovertemplate="<b>MACD</b><br>Ngày: %{x|%d/%m/%Y}<br>Giá trị: %{y:.4f}<extra></extra>"), row=row, col=1)
        if not signal.empty:
            fig.add_trace(go.Scatter(x=signal.index, y=signal.values, mode="lines", name="Signal",
                                     line=dict(color=COLORS["ma_long"], dash="dot", width=1.5), connectgaps=False,
                                     hovertemplate="<b>Signal</b><br>Ngày: %{x|%d/%m/%Y}<br>Giá trị: %{y:.4f}<extra></extra>"), row=row, col=1)
    except Exception as e:
        logger.warning(f"MACD error: {e}")


def _add_stoch_chart(fig, data, row):
    if "Stoch_K" not in data.columns:
        return
    stoch_k = clean_indicator_data(data, "Stoch_K")
    if stoch_k.empty:
        return
    fig.add_trace(go.Scatter(
        x=stoch_k.index, 
        y=stoch_k.values, 
        mode="lines", 
        name="%K",
        line=dict(color=COLORS["stoch"], width=2.0),
        connectgaps=False,
        hovertemplate="<b>%K</b><br>" +
                      "Ngày: %{x|%d/%m/%Y}<br>" +
                      "Giá trị: %{y:.2f}<br>" +
                      "<extra></extra>",
    ), row=row, col=1)
    if "Stoch_D" in data.columns:
        stoch_d = clean_indicator_data(data, "Stoch_D")
        if not stoch_d.empty:
                fig.add_trace(go.Scatter(
                    x=stoch_d.index, 
                    y=stoch_d.values, 
                    mode="lines", 
                    name="%D",
                    line=dict(color=COLORS["ma_long"], dash="dot", width=1.5),
                    connectgaps=False,
                    hovertemplate="<b>%D</b><br>" +
                                  "Ngày: %{x|%d/%m/%Y}<br>" +
                                  "Giá trị: %{y:.2f}<br>" +
                                  "<extra></extra>",
                ), row=row, col=1)
    for y, c in [(STOCH_OVERBOUGHT, COLORS["bearish"]), (STOCH_NEUTRAL, COLORS["neutral"]), (STOCH_OVERSOLD, COLORS["bullish"])]:
        fig.add_hline(y=y, line=dict(color=c, dash="dash", width=1.0), row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=row, col=1)


def _add_adx_chart(fig, data, row):
    if "ADX" not in data.columns:
        return
    adx_data = clean_indicator_data(data, "ADX")
    if adx_data.empty:
        return
    fig.add_trace(go.Scatter(
        x=adx_data.index, 
        y=adx_data.values, 
        mode="lines", 
        name="ADX",
        line=dict(color=COLORS["adx"], width=2.0),
        connectgaps=False,
        hovertemplate="<b>ADX</b><br>" +
                      "Ngày: %{x|%d/%m/%Y}<br>" +
                      "Giá trị: %{y:.2f}<br>" +
                      "<extra></extra>",
    ), row=row, col=1)
    if "ADX_+DI" in data.columns:
        di_plus = clean_indicator_data(data, "ADX_+DI")
        if not di_plus.empty:
            fig.add_trace(go.Scatter(
                x=di_plus.index, 
                y=di_plus, 
                mode="lines", 
                name="+DI",
                line=dict(color=COLORS["bullish"], width=1.5),
                hovertemplate="<b>+DI</b><br>" +
                              "Ngày: %{x|%d/%m/%Y}<br>" +
                              "Giá trị: %{y:.2f}<br>" +
                              "<extra></extra>",
            ), row=row, col=1)
    if "ADX_-DI" in data.columns:
        di_minus = clean_indicator_data(data, "ADX_-DI")
        if not di_minus.empty:
            fig.add_trace(go.Scatter(
                x=di_minus.index, 
                y=di_minus, 
                mode="lines", 
                name="-DI",
                line=dict(color=COLORS["bearish"], width=1.5),
                hovertemplate="<b>-DI</b><br>" +
                              "Ngày: %{x|%d/%m/%Y}<br>" +
                              "Giá trị: %{y:.2f}<br>" +
                              "<extra></extra>",
            ), row=row, col=1)
    for y, c in [(ADX_STRONG_TREND, COLORS["resistance"]), (ADX_WEAK_TREND, COLORS["neutral"])]:
        fig.add_hline(y=y, line=dict(color=c, dash="dash", width=1.0), row=row, col=1)
    fig.update_yaxes(range=[0, 70], row=row, col=1)


def _add_levels(fig, data, levels, row):
    try:
        for lvl, val in (levels or {}).items():
            fig.add_trace(go.Scatter(
                x=[data.index.min(), data.index.max()],
                y=[val, val],
                mode="lines",
                name=lvl,
                line=dict(
                    color=COLORS["support"] if "fib_100" in lvl.lower() or "support" in lvl.lower() else COLORS["resistance"],
                    dash="dash",
                    width=2.0,
                ),
                showlegend=False, 
            ), row=row, col=1)
    except Exception as e:
        logger.warning(f"Levels error: {e}")


def _add_patterns(fig, data, patterns, row):
    try:
        if isinstance(patterns, pd.DataFrame) and not patterns.empty and "Head_and_Shoulders" in patterns.columns:
            pattern_dates = patterns[patterns["Head_and_Shoulders"] == 1].index
            
            # Kiểm tra pattern_dates không rỗng
            if len(pattern_dates) == 0:
                return
            
            # Lấy giá đỉnh tại các điểm pattern
            y_values = data.loc[pattern_dates, "High"].values
            
            fig.add_trace(go.Scatter(
                x=pattern_dates, 
                y=y_values,
                mode="markers+lines", 
                marker=dict(
                    symbol="triangle-up", 
                    size=12, 
                    color=COLORS["pattern"], 
                    line=dict(width=1.5, color='white')
                ),
                line=dict(color=COLORS["pattern"], width=2, dash="dashdot"),
                name="H&S Pattern",
                showlegend=True,
                hovertemplate="<b>H&S Pattern</b><br>" +
                              "Ngày: %{x|%d/%m/%Y}<br>" +
                              "Giá: %{y:,.0f}<br>" +
                              "<extra></extra>",
            ), row=row, col=1)
            
            # Tính neckline thông minh: lấy điểm thấp nhất giữa các đỉnh
            pattern_lows = data.loc[pattern_dates, "Low"]
            if not pattern_lows.empty:
                neckline_price = pattern_lows.min()
                
                fig.add_hline(
                    y=neckline_price, 
                    line=dict(color=COLORS["pattern"], width=2, dash="dash"),
                    annotation_text=f"Neckline: {neckline_price:,.0f}",
                    annotation_position="top left",
                    annotation_font_color=COLORS["pattern"],
                    row=row, col=1
                )
                
    except Exception as e:
        logger.warning(f"Pattern error: {e}")


def _update_dark_layout(fig, title, height, total_rows, data, default_visible_days):
    
    page_background_color = "#0f172a" 
    grid_color = "rgba(255,255,255,0.05)"
    
    # FIXED: Tính toán phạm vi hiển thị thông minh hơn
    if len(data) > default_visible_days:
        # Lấy N điểm dữ liệu cuối cùng thay vì N ngày
        visible_data = data.iloc[-default_visible_days:]
        visible_start = visible_data.index[0]
        visible_end = visible_data.index[-1]
    else:
        visible_start = data.index[0]
        visible_end = data.index[-1]
    
    # Thêm padding 2% để tránh nến bị sát biên
    time_padding = (visible_end - visible_start) * 0.02

    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.5, font=dict(color="#E0E0E0", size=20, weight=600)),
        paper_bgcolor=page_background_color,
        plot_bgcolor=page_background_color,
        font=dict(color="#E0E0E0", size=11),
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=1.10,
            xanchor="right", 
            x=1, 
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1,
            font=dict(size=11)
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(20, 20, 20, 0.97)",
            font_size=12,
            font_color="white",
            bordercolor="rgba(0, 230, 118, 0.5)",
            font_family="Arial"
        ),
        margin=dict(l=55, r=45, t=120, b=70),
        xaxis_rangeslider_visible=False,
        dragmode="zoom",
    )
    
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    
    # Trục Y
    for i in range(1, total_rows + 1):
        y_title = fig.layout.annotations[i-1].text 
        
        fig.update_yaxes(
            showgrid=True, 
            gridcolor=grid_color,
            showticklabels=True,
            title_text=y_title,
            titlefont=dict(size=12, color="#B0BEC5"),
            tickfont=dict(size=10),
            row=i, col=1,
            zeroline=False if i == 1 else True
        )
        
        if i == 1:
            fig.update_yaxes(
                tickformat=".2f", 
                autorange=True,
                rangemode='normal',
                range=[data['Low'].min() * 0.95, data['High'].max() * 1.05] if 'Low' in data.columns and 'High' in data.columns else None,
                row=1, col=1
            )
        elif y_title == "Volume":
            fig.update_yaxes(rangemode='tozero', row=i, col=1)
    
    # FIXED: Trục X - Cải thiện logic
    for i in range(1, total_rows + 1):
        show_labels = (i == total_rows)
        
        # Cấu hình cơ bản cho tất cả subplot
        fig.update_xaxes(
            showgrid=True, 
            gridcolor=grid_color,
            rangeslider_visible=False, 
            showticklabels=show_labels, 
            tickformat="%d/%m/%Y" if show_labels else None,
            tickangle=-45 if show_labels else 0,
            tickmode="auto",
            nticks=15,
            title_text="Ngày" if i == total_rows else None,
            titlefont=dict(size=12, color="#B0BEC5") if i == total_rows else None,
            tickfont=dict(size=9) if show_labels else None,
            type="date",
            # FIXED: Chỉ loại bỏ cuối tuần nếu data dày đặc
            rangebreaks=[
                dict(bounds=["sat", "mon"])
            ] if len(data) > 100 else [],  # Chỉ dùng cho data nhiều
            row=i, col=1
        )
        
        # CHỈ set range cho main chart (row 1)
        if i == 1:
            fig.update_xaxes(
                range=[visible_start - time_padding, visible_end + time_padding],
                row=1, col=1
            )
    
    # Range Selector chỉ cho main chart
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1 tháng", step="month", stepmode="backward"),
                dict(count=3, label="3 tháng", step="month", stepmode="backward"),
                dict(count=6, label="6 tháng", step="month", stepmode="backward"),
                dict(count=1, label="1 năm", step="year", stepmode="backward"),
                dict(step="all", label="Tất cả")
            ]),
            bgcolor='rgba(30, 30, 30, 0.95)',
            font=dict(color="#FFFFFF", size=10),
            activecolor='rgba(0, 230, 118, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1,
            x=0.0,
            y=1.18,
            xanchor='left',
            yanchor='top'
        ),
        row=1, col=1  # FIXED: Chỉ định rõ row 1
    )

    fig.update_annotations(font_size=14, font_color="#E0E0E0")