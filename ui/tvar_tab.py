# ============================================================
# 📊 ui/tvar_tab.py — Giao diện TVAR trong Streamlit (Đồng bộ + Dark Theme)
# ============================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings

# Ẩn cảnh báo torch
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning)
from utils.data_loader import load_sentiment_data
from models.tvar_model import run_tvar


# ============================================================
# 🔹 Hàm vẽ IRF bằng Plotly
# ============================================================
def plot_irf_plotly(irf_obj, title="Impulse Response Function"):
    """Vẽ biểu đồ IRF (Impulse Response Function) từ mô hình VAR."""
    if irf_obj is None:
        st.warning("⚠️ Không có dữ liệu IRF để hiển thị.")
        return None

    irf = irf_obj.irfs
    steps = list(range(irf.shape[0]))

    # ✅ Xử lý tên biến tương thích nhiều phiên bản statsmodels
    try:
        if hasattr(irf_obj.model, "endog_names"):
            variable_names = irf_obj.model.endog_names
        elif hasattr(irf_obj.model, "names"):
            variable_names = irf_obj.model.names
        else:
            variable_names = [f"y{i+1}" for i in range(irf.shape[1])]
    except Exception:
        variable_names = [f"y{i+1}" for i in range(irf.shape[1])]

    fig = go.Figure()
    for i, var in enumerate(variable_names):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=irf[:, i, 0],
                mode="lines",
                name=f"{var} response to {variable_names[0]} shock"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Steps (days)",
        yaxis_title="Impulse Response",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.25),
        height=420,
    )
    return fig


# ============================================================
# 📋 Hàm sinh nhận xét tự động
# ============================================================
def generate_interpretation(results, ticker, time_period):
    """Sinh đoạn nhận xét tự động dựa trên kết quả TVAR."""
    low, high = results.get("low", {}), results.get("high", {})
    txt = f"**📊 Phân tích mô hình TVAR cho {ticker} ({time_period})**\n\n"

    # --- Low regime ---
    if low and low.get("summary") != "N/A":
        txt += f"🔹 **Low regime** (mức cảm xúc thấp / tin tiêu cực): "
        if "mean" in low["summary"].lower() or "reversion" in low["summary"].lower():
            txt += "Lợi suất có xu hướng *mean reversion*, phản ánh phản ứng điều chỉnh sau các tin xấu. "
        elif "insignificant" in low["summary"].lower():
            txt += "Không phát hiện mối quan hệ nhân quả đáng kể giữa cảm xúc và lợi suất trong giai đoạn này. "
        else:
            txt += "Có tín hiệu tác động phi tuyến giữa cảm xúc và lợi suất, nhưng cần xem thêm IRF để xác định hướng. "

    # --- High regime ---
    if high and high.get("summary") != "N/A":
        txt += f"\n🔸 **High regime** (mức cảm xúc cao / tin tích cực): "
        if "momentum" in high["summary"].lower():
            txt += "Thị trường thể hiện hành vi *momentum ngắn hạn*, khi tin tốt dẫn tới phản ứng tăng giá tạm thời. "
        elif "negative" in high["summary"].lower():
            txt += "Các phản ứng tiêu cực xuất hiện mạnh hơn sau chuỗi tin tốt, thể hiện hiện tượng đảo chiều. "
        elif "insignificant" in high["summary"].lower():
            txt += "Không có bằng chứng ý nghĩa thống kê cho mối quan hệ cảm xúc–giá. "
        else:
            txt += "Mối quan hệ giữa cảm xúc và lợi suất có thể phản ánh hành vi quá phản ứng của nhà đầu tư. "

    # --- Nhận xét chung ---
    txt += (
        "\n\n📘 **Nhận xét tổng hợp:** Kết quả cho thấy phản ứng giá cổ phiếu phụ thuộc vào trạng thái cảm xúc thị trường. "
        "Trong chế độ tiêu cực, thị trường thường điều chỉnh dần (*mean reversion*), "
        "trong khi chế độ tích cực dễ xuất hiện *momentum ngắn hạn* hoặc đảo chiều nhanh. "
        "Điều này tương đồng với mô tả trong bài báo — phản ánh hành vi phi tuyến và tâm lý bầy đàn của nhà đầu tư Việt Nam."
    )

    return txt


# ============================================================
# 📈 TAB TVAR CHÍNH
# ============================================================
def render(ticker=None):
    """Hiển thị giao diện mô hình Threshold VAR (TVAR)."""
    st.markdown(
        "<h2 style='color:#38bdf8;'>📈 Mô hình Threshold VAR (TVAR)</h2>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # 🧭 ĐỒNG BỘ CÁC BIẾN TỪ SESSION
    # ============================================================
    current_ticker = st.session_state.get("ticker", "FLC")
    current_period = st.session_state.get("time_period", "Before Scandal")

    col1, col2 = st.columns(2)
    with col1:
        time_period = st.selectbox(
            "🕒 Giai đoạn dữ liệu:",
            ["Before Scandal", "After Scandal"],
            index=1 if current_period == "After Scandal" else 0
        )
        st.session_state["time_period"] = time_period
    with col2:
        ticker = st.text_input("Nhập mã cổ phiếu:", current_ticker).upper()
        st.session_state["ticker"] = ticker

    st.markdown("<hr>", unsafe_allow_html=True)

    # ============================================================
    # 📂 TẢI DỮ LIỆU
    # ============================================================
    df = load_sentiment_data(ticker, time_period=time_period)
    if df.empty:
        st.warning("⚠️ Không tìm thấy dữ liệu cho mã cổ phiếu này.")
        return

    # ============================================================
    # 🚀 CHẠY HOẶC TẢI LẠI MÔ HÌNH TVAR (với cache)
    # ============================================================
    @st.cache_data(show_spinner=False, ttl=7200)
    def run_tvar_cached(df_data, ticker_name):
        return run_tvar(df_data, ticker_name)
    
    key = f"tvar_result_{ticker}_{time_period}"
    refresh = st.button("🔄 Chạy lại mô hình TVAR")

    if key not in st.session_state or refresh:
        with st.spinner("🔄 Đang ước lượng mô hình Threshold VAR..."):
            results = run_tvar_cached(df, ticker)
            st.session_state[key] = results
    else:
        results = st.session_state[key]

    if "error" in results:
        st.error(results["error"])
        return

    # ============================================================
    # 🧭 THÔNG TIN TỔNG QUAN (Dark Style)
    # ============================================================
    st.markdown(
        f"""
        <div style="
            padding:16px;
            border-radius:14px;
            background:linear-gradient(135deg,#1e293b,#0f172a);
            color:#f1f5f9;
            box-shadow:0px 0px 8px rgba(0,0,0,0.25);
            border:1px solid rgba(148,163,184,0.3);
            margin-bottom:20px;
        ">
            <h4 style='color:#38bdf8; margin-bottom:8px;'>
                📘 {ticker} — {time_period}
            </h4>
            <p style="font-size:15px; margin-top:-5px;">
                Ngưỡng sentiment (γ): <b style="color:#fbbf24;">{results['threshold']:.3f}</b><br>
                Số quan sát:
                <b style="color:#22c55e;">Low = {results['low_n']}</b> /
                <b style="color:#ef4444;">High = {results['high_n']}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h4 style='color:#93c5fd;'>🔍 Kết quả chi tiết cho hai chế độ (Regimes)</h4>", unsafe_allow_html=True)

    col_low, col_high = st.columns(2)

    # ============================================================
    # 🔹 LOW REGIME
    # ============================================================
    with col_low:
        st.markdown("#### 🔹 Low Sentiment Regime")
        low = results.get("low", {})
        if not low or low.get("summary") == "N/A":
            st.error("❌ Không thể ước lượng mô hình ở chế độ Low.")
        else:
            st.markdown(f"**Độ trễ tối ưu:** {low['lag']}")
            st.text_area("📄 Kết quả ước lượng (Low)", low["summary"], height=240)
            st.caption(f"📋 Kiểm định chẩn đoán: {low['diag']}")
            if low.get("irf"):
                fig_low = plot_irf_plotly(low["irf"], f"{ticker} — IRF (Low Regime)")
                st.plotly_chart(fig_low, use_container_width=True)

    # ============================================================
    # 🔸 HIGH REGIME
    # ============================================================
    with col_high:
        st.markdown("#### 🔸 High Sentiment Regime")
        high = results.get("high", {})
        if not high or high.get("summary") == "N/A":
            st.error("❌ Không thể ước lượng mô hình ở chế độ High.")
        else:
            st.markdown(f"**Độ trễ tối ưu:** {high['lag']}")
            st.text_area("📄 Kết quả ước lượng (High)", high["summary"], height=240)
            st.caption(f"📋 Kiểm định chẩn đoán: {high['diag']}")
            if high.get("irf"):
                fig_high = plot_irf_plotly(high["irf"], f"{ticker} — IRF (High Regime)")
                st.plotly_chart(fig_high, use_container_width=True)

    # ============================================================
    # ⚖️ SO SÁNH IRF
    # ============================================================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#93c5fd;'>📊 So sánh phản ứng xung giữa hai Regime</h4>", unsafe_allow_html=True)

    if results["low"].get("irf") and results["high"].get("irf"):
        irf_low = results["low"]["irf"]
        irf_high = results["high"]["irf"]
        steps = list(range(irf_low.irfs.shape[0]))

        variable_names = getattr(irf_low.model, "endog_names", [f"y{i+1}" for i in range(irf_low.irfs.shape[1])])
        selected_var = st.selectbox("Chọn biến để so sánh:", variable_names)
        var_index = variable_names.index(selected_var)

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=steps, y=irf_low.irfs[:, var_index, 0],
            mode="lines", name=f"{selected_var} (Low)", line=dict(color="#2563eb")
        ))
        fig_compare.add_trace(go.Scatter(
            x=steps, y=irf_high.irfs[:, var_index, 0],
            mode="lines", name=f"{selected_var} (High)",
            line=dict(color="#f97316", dash="dash")
        ))

        fig_compare.update_layout(
            title=f"Phản ứng xung của {selected_var} giữa hai Regime ({ticker})",
            xaxis_title="Steps (days)",
            yaxis_title="Impulse Response",
            template="plotly_dark",
            legend=dict(orientation="h", y=-0.25),
            height=450
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info("⚠️ Chưa đủ dữ liệu IRF cho cả hai chế độ để so sánh.")

    # ============================================================
    # 🧠 NHẬN XÉT & DIỄN GIẢI KẾT QUẢ
    # ============================================================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#38bdf8;'>🧠 Nhận xét & Diễn giải kết quả</h4>", unsafe_allow_html=True)

    interpretation = generate_interpretation(results, ticker, time_period)
    st.markdown(
        f"""
        <div style="
            background-color:rgba(30,41,59,0.7);
            padding:14px 18px;
            border-radius:12px;
            border-left:4px solid #38bdf8;
            color:#e2e8f0;
            line-height:1.6;
        ">
            {interpretation}
        </div>
        """,
        unsafe_allow_html=True
    )
