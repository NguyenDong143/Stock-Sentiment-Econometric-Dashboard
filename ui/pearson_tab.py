import streamlit as st
import pandas as pd
from scipy.stats import pearsonr
from utils.data_loader import load_sentiment_data

# =============================
# 📘 KẾT QUẢ NGHIÊN CỨU CHÍNH THỨC
# =============================
PEARSON_RESULTS = {
    "Before Scandal": {
        "AMD": {"r": 0.196535, "p-value": 0.000000, "Kết luận": "Tương quan dương, có ý nghĩa thống kê"},
        "ART": {"r": 0.252156, "p-value": 0.000000, "Kết luận": "Tương quan dương, có ý nghĩa thống kê"},
        "FLC": {"r": 0.245598, "p-value": 0.000000, "Kết luận": "Tương quan dương, có ý nghĩa thống kê"},
        "GAB": {"r": -0.093008, "p-value": 0.006787, "Kết luận": "Tương quan âm yếu, có ý nghĩa thống kê"},
        "HAI": {"r": 0.175241, "p-value": 0.000000, "Kết luận": "Tương quan dương, có ý nghĩa thống kê"},
    },
    "After Scandal": {
        "AMD": {"r": 0.128505, "p-value": 0.005887, "Kết luận": "Tương quan dương, có ý nghĩa thống kê"},
        "ART": {"r": 0.147826, "p-value": 0.001512, "Kết luận": "Tương quan dương, có ý nghĩa thống kê"},
        "FLC": {"r": 0.135791, "p-value": 0.003596, "Kết luận": "Tương quan dương, có ý nghĩa thống kê"},
        "GAB": {"r": -0.002622, "p-value": 0.955368, "Kết luận": "Không có tương quan"},
        "HAI": {"r": 0.117182, "p-value": 0.012087, "Kết luận": "Tương quan dương, có ý nghĩa thống kê"},
    },
}


def render(ticker: str = None):
    st.header("📊 Kiểm định Tương quan (Pearson)")

    # --- Lấy thông tin cấu hình ---
    data_type = st.session_state.get("data_type", "Content")
    time_period = st.session_state.get("time_period", "Before Scandal")

    st.markdown(
        f"<p style='color:#94a3b8'>Dữ liệu: <b>{data_type}</b> | Giai đoạn: <b>{time_period}</b></p>",
        unsafe_allow_html=True,
    )

    # ======================================================
    # 🧾 KẾT QUẢ NGHIÊN CỨU CHÍNH THỨC (nếu có)
    # ======================================================
    if ticker in PEARSON_RESULTS.get(time_period, {}):
        st.subheader("📘 Kết quả thực nghiệm chính thức (2018–2023)")
        res = PEARSON_RESULTS[time_period][ticker]
        df_show = pd.DataFrame([res])
        st.table(df_show)

        r, p = res["r"], res["p-value"]
        if p < 0.05:
            st.success(f"✅ Có tương quan có ý nghĩa thống kê (p = {p:.4f}). Hệ số r = {r:.3f}.")
        else:
            st.warning(f"⚠️ Không phát hiện tương quan đáng kể (p = {p:.4f}).")
        st.divider()

    # ======================================================
    # ⚙️ KIỂM ĐỊNH THỰC TẾ (tuỳ chọn)
    # ======================================================
    st.subheader("⚙️ Tùy chọn: Thực hiện kiểm định Pearson thực tế")

    run_test = st.toggle("Chạy kiểm định Pearson thực tế (Python)", value=False)
    if not run_test:
        st.caption("🔹 Bật tùy chọn này để kiểm tra dữ liệu và chạy kiểm định thực tế.")
        return

    # --- Tải dữ liệu ---
    df = load_sentiment_data(ticker, data_type, time_period)
    if df.empty:
        st.warning("⚠️ Không tìm thấy dữ liệu để kiểm định.")
        return

    # --- Kiểm tra cột bắt buộc ---
    if "label" not in df.columns:
        st.error("❌ Thiếu cột 'label' trong dữ liệu (điểm cảm xúc).")
        return

    price_col = next((c for c in df.columns if c.lower() in ["close", "adj close"]), None)
    if not price_col:
        st.error("❌ Thiếu cột 'close' hoặc 'adj close' trong dữ liệu (giá cổ phiếu).")
        return

    # --- Làm sạch dữ liệu ---
    df = df.dropna(subset=["label", price_col])
    df = df.sort_values("date")

    if len(df) < 5:
        st.warning("⚠️ Dữ liệu không đủ để thực hiện kiểm định (ít hơn 5 quan sát).")
        return

    # --- Kiểm định Pearson với cache ---
    @st.cache_data(show_spinner=False, ttl=7200)
    def compute_pearson(label_data, price_data):
        return pearsonr(label_data, price_data)
    
    corr, pval = compute_pearson(df["label"].values, df[price_col].values)

    col1, col2 = st.columns(2)
    col1.metric("📈 Hệ số tương quan (r)", f"{corr:.3f}")
    col2.metric("📊 P-value", f"{pval:.4f}")

    if pval < 0.05:
        if corr > 0:
            st.success("✅ Có mối quan hệ tuyến tính **tích cực** có ý nghĩa thống kê (p < 0.05).")
        else:
            st.success("✅ Có mối quan hệ tuyến tính **tiêu cực** có ý nghĩa thống kê (p < 0.05).")
    else:
        st.warning("⚠️ Không có mối tương quan đáng kể (p ≥ 0.05).")

    # ======================================================
    # 🎨 BIỂU ĐỒ TRỰC QUAN
    # ======================================================
    st.subheader("📉 Biểu đồ tương quan")

    tab1, tab2 = st.tabs(["📊 Phân phối theo nhóm cảm xúc", "📈 Giá trung bình theo cảm xúc"])

    with tab1:
        @st.cache_data(show_spinner=False, ttl=3600)
        def create_strip_plot(df_data, price_column, ticker_name):
            import plotly.express as px
            df_plot = df_data.copy()
            df_plot["label"] = df_plot["label"].astype("category")
            fig = px.strip(
                df_plot,
                x="label",
                y=price_column,
                color="label",
                title=f"Phân phối giá cổ phiếu theo cảm xúc ({ticker_name})",
                labels={"label": "Nhóm cảm xúc", price_column: "Giá cổ phiếu"},
                stripmode="overlay",
            )
            fig.update_traces(opacity=0.6, jitter=0.35)
            fig.update_layout(showlegend=False)
            return fig
        
        fig1 = create_strip_plot(df, price_col, ticker.upper())
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        @st.cache_data(show_spinner=False, ttl=3600)
        def create_bar_plot(df_data, price_column, ticker_name):
            import plotly.express as px
            avg_price = df_data.groupby("label")[price_column].mean().reset_index()
            fig = px.bar(
                avg_price,
                x="label",
                y=price_column,
                color="label",
                text=price_column,
                title=f"Giá cổ phiếu trung bình theo cảm xúc ({ticker_name})",
                labels={"label": "Nhóm cảm xúc", price_column: "Giá trung bình"},
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                hovertemplate="Cảm xúc: %{x}<br>Giá TB: %{y:.2f}",
            )
            fig.update_layout(showlegend=False, yaxis_title="Giá cổ phiếu (VNĐ)")
            return fig
        
        fig2 = create_bar_plot(df, price_col, ticker.upper())
        st.plotly_chart(fig2, use_container_width=True)

    # ======================================================
    # 🧾 DỮ LIỆU GẦN NHẤT
    # ======================================================
    st.subheader("🧾 Dữ liệu gần nhất")
    cols_to_show = [c for c in ["date", "label", price_col] if c in df.columns]
    st.dataframe(df[cols_to_show].tail(10), use_container_width=True)
