import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List


# =======================================================
# 1️⃣ BIỂU ĐỒ PHÂN BỐ CẢM XÚC (Histogram)
# =======================================================
def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Hiển thị phân bố cảm xúc tin tức theo cột 'label':
    - 1: Tích cực
    - 0: Trung tính
    - -1: Tiêu cực
    """
    if "label" not in df.columns:
        # Giả định cột 'label' chứa giá trị -1, 0, 1
        raise ValueError("⚠️ Thiếu cột 'label' trong DataFrame (cần cho Phân bố Cảm xúc)!")

    # Ánh xạ giá trị cảm xúc
    mapping = {-1: "Tiêu cực 😞", 0: "Trung tính 😐", 1: "Tích cực 😃"}
    df["Sentiment_Label"] = df["label"].map(mapping)

    color_map = {
        "Tích cực 😃": "#10b981",    # Xanh lá cây (Emerald)
        "Trung tính 😐": "#3b82f6",  # Xanh dương (Blue)
        "Tiêu cực 😞": "#ef4444"     # Đỏ (Red)
    }

    fig = px.histogram(
        df,
        x="Sentiment_Label",
        color="Sentiment_Label",
        title="📊 Phân bố cảm xúc tin tức (PhoBERT)",
        barmode="group",
        color_discrete_map=color_map,
        height=400
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Cảm xúc",
        yaxis_title="Số lượng tin",
        showlegend=False,
        xaxis={'categoryorder': 'array', 'categoryarray': ['Tích cực 😃', 'Trung tính 😐', 'Tiêu cực 😞']}
    )
    return fig


# =======================================================
# 2️⃣ BIỂU ĐỒ DONUT CẢM XÚC
# =======================================================
def plot_sentiment_donut(df: pd.DataFrame) -> go.Figure:
    """
    Biểu đồ donut thể hiện tỷ lệ cảm xúc dựa trên cột 'label'
    """
    if "label" not in df.columns:
        raise ValueError("⚠️ Thiếu cột 'label' trong DataFrame (cần cho Donut Chart)!")

    mapping = {-1: "Tiêu cực 😞", 0: "Trung tính 😐", 1: "Tích cực 😃"}
    df["Sentiment_Label"] = df["label"].map(mapping)

    counts = df["Sentiment_Label"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    color_map = {
        "Tích cực 😃": "#10b981",
        "Trung tính 😐": "#3b82f6",
        "Tiêu cực 😞": "#ef4444"
    }

    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        hole=0.5,
        title="🧭 Tỷ lệ cảm xúc tin tức (Donut Chart)",
        color="Sentiment",
        color_discrete_map=color_map,
        height=400
    )
    fig.update_layout(template="plotly_dark", showlegend=True, margin=dict(t=50, b=20, l=20, r=20))
    return fig


# =======================================================
# 3️⃣ BIỂU ĐỒ GIÁ CỔ PHIẾU (SỬ DỤNG CỘT CHUẨN: date, close)
# =======================================================
def plot_price_trend(df: pd.DataFrame, ticker: str = "Cổ phiếu") -> go.Figure:
    """
    Hiển thị diễn biến giá cổ phiếu theo thời gian.
    Sử dụng cột 'date' và 'close' (đã chuẩn hóa từ data_loader).
    """
    if not {"date", "close"}.issubset(df.columns):
        raise ValueError("⚠️ Thiếu cột 'date' hoặc 'close' trong dữ liệu giá!")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        mode="lines", name="Giá đóng cửa",
        line=dict(color="#38bdf8", width=2) # Sky blue
    ))
    fig.update_layout(
        title=f"📈 Diễn biến giá cổ phiếu {ticker}",
        template="plotly_dark",
        xaxis_title="Ngày",
        yaxis_title="Giá đóng cửa (VND)",
        height=500,
        hovermode="x unified"
    )
    return fig


# =======================================================
# 4️⃣ BIỂU ĐỒ TƯƠNG QUAN PEARSON
# =======================================================
def plot_corr_scatter(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """
    Biểu đồ scatter thể hiện mối tương quan giữa 2 biến, có đường hồi quy OLS.
    """
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"⚠️ Thiếu cột '{x}' hoặc '{y}' trong dữ liệu!")

    fig = px.scatter(
        df,
        x=x,
        y=y,
        trendline="ols",
        title=f"Tương quan giữa {x} và {y}",
        opacity=0.7,
        color_discrete_sequence=["#38bdf8"],
        height=500
    )
    fig.update_layout(template="plotly_dark", xaxis_title=x, yaxis_title=y)
    return fig


# =======================================================
# 5️⃣ HEATMAP MA TRẬN TƯƠNG QUAN
# =======================================================
def plot_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap hiển thị ma trận tương quan giữa các biến số.
    """
    # Tính tương quan
    corr = df.corr(numeric_only=True)
    
    # Tạo text cho ô vuông (nếu muốn hiển thị số)
    text = corr.applymap(lambda x: f'{x:.2f}' if pd.notna(x) else '')
    
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r", # Đỏ-Xanh (Red-Blue reversed)
        title="🌡️ Ma trận tương quan giữa các biến",
        text_auto=".2f",
        aspect="auto",
        height=600
    )
    fig.update_layout(template="plotly_dark")
    return fig


# =======================================================
# 6️⃣ BIỂU ĐỒ P-VALUE THEO LAG (CHO GRANGER)
# =======================================================
def plot_pvalue_bars(p_values: Dict[int, float]) -> go.Figure:
    """
    Biểu đồ thanh (bar chart) thể hiện p-value theo từng độ trễ (lag)
    """
    if not p_values:
        raise ValueError("⚠️ Không có kết quả p-value để hiển thị!")

    # Chuyển đổi Dict sang DataFrame
    df = pd.DataFrame(list(p_values.items()), columns=["Lag", "p_value"])
    # Xác định mức ý nghĩa thống kê
    df["Significant"] = df["p_value"] < 0.05

    fig = px.bar(
        df,
        x="Lag",
        y="p_value",
        color="Significant",
        color_discrete_map={True: "#10b981", False: "#6b7280"},
        title="P-value theo độ trễ (Kiểm định Granger)",
        height=500
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Ngưỡng 0.05")
    fig.update_layout(
        template="plotly_dark", 
        yaxis_title="p-value", 
        xaxis_title="Lag",
        yaxis_range=[0, 1.0] # Đảm bảo trục Y luôn hiển thị từ 0 đến 1
    )
    return fig


# =======================================================
# 7️⃣ XU HƯỚNG CẢM XÚC THEO THỜI GIAN (LÀM MƯỢT DỮ LIỆU)
# =======================================================
def plot_sentiment_over_time(df: pd.DataFrame, window: int = 14) -> Optional[go.Figure]:
    """
    Vẽ biểu đồ xu hướng cảm xúc theo thời gian (làm mượt bằng rolling mean).
    """
    # 🔹 Chuẩn hóa cột ngày (Mong đợi cột 'date' đã được chuẩn hóa)
    if 'date' not in df.columns:
        st.warning("⚠️ Thiếu cột 'date' trong dữ liệu! Không thể vẽ xu hướng cảm xúc.")
        return None
        
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df = df.dropna(subset=['date']).sort_values('date')

    # 🔹 Kiểm tra cột cảm xúc
    required_cols = ["tích cực", "trung tính", "tiêu cực"]
    if not all(col in df.columns for col in required_cols):
        st.warning("⚠️ Thiếu các cột 'tích cực', 'trung tính', 'tiêu cực' trong dữ liệu!")
        return None

    # 🔹 Tính trung bình theo ngày
    daily = df.groupby('date')[required_cols].mean().reset_index()

    # 🔹 Làm mượt bằng rolling mean
    for col in required_cols:
         daily[f"{col}_smooth"] = daily[col].rolling(window=window, min_periods=1).mean()

    # 🔹 Vẽ biểu đồ
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily["tích cực_smooth"],
        mode="lines", name=f"Tích cực 😃 (TB {window} ngày)",
        line=dict(color="#10b981", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily["trung tính_smooth"],
        mode="lines", name=f"Trung tính 😐 (TB {window} ngày)",
        line=dict(color="#3b82f6", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily["tiêu cực_smooth"],
        mode="lines", name=f"Tiêu cực 😞 (TB {window} ngày)",
        line=dict(color="#ef4444", width=2)
    ))

    fig.update_layout(
        title=f"🕒 Xu hướng cảm xúc theo thời gian (Smooth)",
        xaxis_title="Ngày",
        yaxis_title="Điểm cảm xúc trung bình (rolling mean)",
        template="plotly_dark",
        hovermode="x unified",
        title_x=0.5,
        legend_title="Nhóm cảm xúc",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# =======================================================
# 8️⃣ BIỂU ĐỒ KDE (PHÂN PHỐI MỨC ĐỘ CẢM XÚC)
# =======================================================
@st.cache_data(show_spinner=False, ttl=3600)
def plot_sentiment_kde(df: pd.DataFrame):
    """
    Biểu đồ KDE phong cách TradingView với LEGEND tách sang bên phải.
    """

    POS_COLOR = "#10B981"
    NEU_COLOR = "#3B82F6"
    NEG_COLOR = "#EF4444"

    TEXT_COLOR = "#E5E7EB"
    GRID_COLOR = (1, 1, 1, 0.06)
    FACE_BG = "none"

    required_cols = ["tích cực", "tiêu cực", "trung tính"]
    if not all(c in df.columns for c in required_cols):
        st.warning("⚠️ Thiếu cột cảm xúc!")
        return

    # Nếu muốn legend nằm ngoài → cần tăng width figure
    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=120)
    fig.patch.set_facecolor(FACE_BG)
    ax.set_facecolor(FACE_BG)

    # KDE
    try:
        sns.kdeplot(df["tích cực"], ax=ax,
                    color=POS_COLOR, linewidth=2,
                    fill=True, alpha=0.18, label="Tích cực 😃")

        sns.kdeplot(df["trung tính"], ax=ax,
                    color=NEU_COLOR, linewidth=2,
                    fill=True, alpha=0.18, label="Trung tính 😐")

        sns.kdeplot(df["tiêu cực"], ax=ax,
                    color=NEG_COLOR, linewidth=2,
                    fill=True, alpha=0.18, label="Tiêu cực 😞")

    except np.linalg.LinAlgError:
        st.warning("Không đủ dữ liệu để vẽ KDE.")
        return

    # Title
    ax.set_title("Phân phối mức độ cảm xúc (KDE)",
                 color=TEXT_COLOR, fontsize=11,
                 fontweight="bold", pad=6)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=7, pad=2)
    ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=7, pad=2)

    # Grid & spine
    ax.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.5)

    # --------------- 🔥 LEGEND BÊN PHẢI (GIỐNG PLOTLY) ----------------
    legend = ax.legend(
        title="Nhóm cảm xúc",
        fontsize=8,
        title_fontsize=8,
        labelcolor=TEXT_COLOR,
        facecolor=(0.10, 0.12, 0.16, 0.7),
        edgecolor=GRID_COLOR,
        framealpha=0.7,
        borderpad=0.6,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)   # đẩy legend ra ngoài
    )
    plt.setp(legend.get_title(), color=TEXT_COLOR)

    # Tự động chỉnh layout để không bị cắt
    plt.tight_layout(pad=0.5)

    st.pyplot(fig)
    plt.close(fig)


# =======================================================
# 9️⃣ BIỂU ĐỒ IMPULSE RESPONSE FUNCTION (IRF - CHO TVAR)
# =======================================================
def plot_irf(irf_results: pd.DataFrame, shock_var: str, response_var: str, title: str = "Impulse Response Function") -> go.Figure:
    """
    Vẽ biểu đồ Impulse Response Function (IRF) từ kết quả TVAR/VAR.
    Dữ liệu cần có các cột 'Horizon', 'Mean', 'Lower', 'Upper'.
    """
    if irf_results.empty or not {'Horizon', 'Mean', 'Lower', 'Upper'}.issubset(irf_results.columns):
        raise ValueError("⚠️ Dữ liệu IRF bị thiếu hoặc không đúng định dạng (cần Horizon, Mean, Lower, Upper).")
    
    fig = go.Figure()

    # Mean response
    fig.add_trace(go.Scatter(
        x=irf_results['Horizon'],
        y=irf_results['Mean'],
        mode='lines',
        name='Mean Response',
        line=dict(color='#10b981', width=3)
    ))

    # Confidence Interval (Shaded area) - Dải tin cậy
    fig.add_trace(go.Scatter(
        x=irf_results['Horizon'],
        y=irf_results['Upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=irf_results['Horizon'],
        y=irf_results['Lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(16, 185, 129, 0.2)', # Màu xanh lá mờ
        line=dict(width=0),
        name='95% Confidence Interval',
    ))

    # Line at zero
    fig.add_hline(y=0, line_dash="dash", line_color="#facc15", line_width=1) 
    
    fig.update_layout(
        title=f"🚀 {title}: Shock từ {shock_var} tới {response_var}",
        xaxis_title="Horizon (Độ trễ)",
        yaxis_title="Phản ứng tích lũy",
        template="plotly_dark",
        height=500
    )
    return fig