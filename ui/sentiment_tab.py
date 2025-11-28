import streamlit as st
import pandas as pd

from utils.data_loader import load_sentiment_data
from utils.visualization import (
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    plot_sentiment_kde,
)

# ======================================================
# 💬 SENTIMENT ANALYSIS TAB
# ======================================================
def render(ticker: str = None):
    """
    Tab phân tích cảm xúc tin tức bằng PhoBERT.
    Hiển thị phân phối nhãn cảm xúc, xu hướng theo thời gian và KDE plot.
    """

    st.markdown(
        """
        <h3 style='color:#3b82f6'>💬 Phân tích cảm xúc tin tức</h3>
        """,
        unsafe_allow_html=True,
    )

    # ==============================
    # 🤖 PHOBERT DEMO – PHÂN TÍCH CẢM XÚC MỚI
    # ==============================
    st.subheader("🤖 PHOBERT DEMO – PHÂN TÍCH CẢM XÚC MỚI")

    text_input = st.text_area(
        "Nhập đoạn tin tức hoặc tiêu đề cần phân tích:", 
        height=120,
        placeholder="Ví dụ: Cổ phiếu FLC tăng mạnh sau thông tin tái cấu trúc doanh nghiệp..."
    )

    if st.button("🔍 Phân tích cảm xúc"):
        if text_input.strip():
            try:
                # Lazy import chỉ khi cần phân tích
                from models.sentiment_phobert import classify_sentiment
                with st.spinner("Đang phân tích với PhoBERT..."):
                    labels = classify_sentiment([text_input])
                label_map = {-1: "Tiêu cực 😞", 0: "Trung tính 😐", 1: "Tích cực 😃"}
                sentiment_label = label_map.get(labels[0], "Không xác định")

                # Hiển thị kết quả với màu tương ứng
                color_map = {
                    "Tích cực 😃": "#10b981",
                    "Trung tính 😐": "#3b82f6",
                    "Tiêu cực 😞": "#ef4444"
                }
                color = color_map.get(sentiment_label, "#94a3b8")

                st.markdown(
                    f"""
                    <div style='background-color:{color}22; padding:15px; border-radius:10px;'>
                        <h4 style='color:{color}'>🧠 Kết quả: {sentiment_label}</h4>
                        <p style='color:#e2e8f0'>📰 {text_input}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except ValueError as e:
                st.error(f"⚠️ Dữ liệu đầu vào không hợp lệ: {e}")
            except Exception as e:
                st.error(f"⚠️ Lỗi khi chạy PhoBERT: {e}")
        else:
            st.warning("⚠️ Vui lòng nhập nội dung tin tức để phân tích!")

    # ==============================
    # 📂 FILE UPLOAD SECTION
    # ==============================
    st.markdown("---")
    st.subheader("📂 Tải file tin tức mới để phân tích hàng loạt")

    uploaded = st.file_uploader("Chọn file .xlsx hoặc .csv", type=["xlsx", "csv"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded, encoding="utf-8-sig")
            else:
                df_upload = pd.read_excel(uploaded)

            if "Headline" not in df_upload.columns:
                st.error("❌ File cần có cột 'Headline' chứa nội dung tin tức.")
            else:
                # Lazy import PhoBERT
                from models.sentiment_phobert import classify_sentiment
                with st.spinner("🔄 Đang phân tích cảm xúc bằng PhoBERT..."):
                    results = classify_sentiment(df_upload["Headline"].tolist())

                df_upload["Predicted_Label"] = [
                    {-1: "Tiêu cực 😞", 0: "Trung tính 😐", 1: "Tích cực 😃"}[i] for i in results
                ]

                st.success("✅ Hoàn tất phân tích cảm xúc hàng loạt!")
                st.dataframe(df_upload.head(10), use_container_width=True)

                # Tải kết quả về
                csv = df_upload.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 Tải xuống kết quả (CSV)",
                    csv,
                    file_name="phoBERT_sentiment_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"⚠️ Lỗi khi xử lý file: {e}")

    # ==============================
    # PHẦN PHÂN TÍCH DỮ LIỆU THEO TICKER
    # ==============================
    st.markdown("---")
    st.markdown("---")
    st.markdown(
        """
        <h3 style='color:#3b82f6'>📊 Phân tích dữ liệu cảm xúc theo mã cổ phiếu</h3>
        """,
        unsafe_allow_html=True,
    )

    # Kiểm tra ticker có được chọn không
    if not ticker:
        st.info("ℹ️ Vui lòng chọn mã cổ phiếu từ sidebar để xem phân tích chi tiết.")
        return

    # --- Lấy cấu hình từ sidebar ---
    data_type = st.session_state.get("data_type", "Content")
    time_period = st.session_state.get("time_period", "Before Scandal")

    st.markdown(
        f"""
        <p style='color:#94a3b8'>
        Mã CP: <b>{ticker}</b> | Dữ liệu: <b>{data_type}</b> | Giai đoạn: <b>{time_period}</b>
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ==============================
    # 1️⃣ TẢI DỮ LIỆU
    # ==============================
    df = load_sentiment_data(ticker, data_type, time_period)

    # 🔹 Fix lỗi Arrow serialization (PyArrow): ép kiểu cột object -> string
    if not df.empty:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)
        df = df.convert_dtypes()

    if df.empty:
        st.warning("⚠️ Không tìm thấy dữ liệu cảm xúc phù hợp.")
        return

    # ==============================
    # 2️⃣ TỔNG QUAN DỮ LIỆU + NHẬN XÉT TỰ ĐỘNG
    # ==============================
    st.subheader("📘 Tổng quan dữ liệu cảm xúc")

    n_records = len(df)

    # 🔹 Nếu có cột 'label' (dạng -1, 0, 1)
    if "label" in df.columns:
        mapping = {-1: "Tiêu cực 😞", 0: "Trung tính 😐", 1: "Tích cực 😃"}
        labels = df["label"].map(mapping).value_counts(normalize=True) * 100
        most_common_label = labels.idxmax() if not labels.empty else "N/A"
    else:
        labels = pd.Series()
        most_common_label = "N/A"

    # 🔹 Tính điểm cảm xúc trung bình (dựa trên 3 cột cảm xúc)
    if all(col in df.columns for col in ["tích cực", "trung tính", "tiêu cực"]):
        avg_score = (
            (df["tích cực"].mean() + df["trung tính"].mean() + df["tiêu cực"].mean()) / 3
        )
    else:
        avg_score = None

    col1, col2 = st.columns(2)
    col1.metric("🧾 Số lượng bài viết", f"{n_records:,}")
    col2.metric("📊 Nhãn phổ biến nhất", most_common_label)

    df_sentiment = load_sentiment_data(ticker, data_type, time_period)

    if df_sentiment.empty:
        st.warning("⚠️ Không có dữ liệu cảm xúc để hiển thị.")
    else:
        st.dataframe(df_sentiment.head(10))
        st.caption(f"📊 Tổng số bản ghi: {len(df_sentiment):,}")

    # ==============================
    # 🧠 NHẬN XÉT TỰ ĐỘNG + BẢNG CHI TIẾT
    # ==============================
    if not labels.empty:
        pos = labels.get("Tích cực 😃", 0)
        neu = labels.get("Trung tính 😐", 0)
        neg = labels.get("Tiêu cực 😞", 0)

        # --- Nhận định tự động ---
        comment = "💡 **Nhận định nhanh:** "
        if pos > neg and pos > neu:
            comment += f"Cảm xúc **tích cực chiếm ưu thế** ({pos:.1f}%), "
            if neg > 20:
                comment += f"nhưng vẫn tồn tại {neg:.1f}% tin tiêu cực."
            else:
                comment += f"trong khi tiêu cực chỉ chiếm {neg:.1f}%."
        elif neg > pos and neg > neu:
            comment += f"Cảm xúc **tiêu cực nổi trội** ({neg:.1f}%), phản ánh tâm lý bi quan trên thị trường."
        else:
            comment += f"Cảm xúc **trung tính chiếm ưu thế** ({neu:.1f}%), thể hiện sự ổn định trong tin tức."

        st.markdown("---")
        st.markdown(
            f"""
            <div style='background-color:#1e293b; padding:10px; border-radius:10px;'>
            {comment}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Hiển thị bảng chi tiết tỷ lệ ---
        st.markdown("#### 📊 Chi tiết tỷ lệ cảm xúc")
        st.markdown(
            f"""
            <div style='padding: 8px 0;'>
                <b style='color:#10b981;'>🟢 Tích cực:</b> {pos:.2f}% 
                <div style='background-color:#1e293b; height:10px; border-radius:5px;'>
                    <div style='width:{pos}%; height:10px; background-color:#10b981; border-radius:5px;'></div>
                </div>
            </div>

            <div style='padding: 8px 0;'>
                <b style='color:#3b82f6;'>🔵 Trung tính:</b> {neu:.2f}% 
                <div style='background-color:#1e293b; height:10px; border-radius:5px;'>
                    <div style='width:{neu}%; height:10px; background-color:#3b82f6; border-radius:5px;'></div>
                </div>
            </div>

            <div style='padding: 8px 0;'>
                <b style='color:#ef4444;'>🔴 Tiêu cực:</b> {neg:.2f}% 
                <div style='background-color:#1e293b; height:10px; border-radius:5px;'>
                    <div style='width:{neg}%; height:10px; background-color:#ef4444; border-radius:5px;'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==============================
    # 3️⃣ PHÂN PHỐI CẢM XÚC (BAR CHART)
    # ==============================
    st.subheader("🎯 Phân phối các loại cảm xúc")
    try:
        fig_dist = plot_sentiment_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ phân phối cảm xúc: {e}")

    # ==============================
    # 4️⃣ XU HƯỚNG CẢM XÚC THEO THỜI GIAN
    # ==============================
    st.subheader("🕒 Xu hướng cảm xúc theo thời gian")
    try:
        fig_time = plot_sentiment_over_time(df)
        if fig_time:
            st.plotly_chart(fig_time, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ xu hướng cảm xúc: {e}")

    # ==============================
    # 5️⃣ KDE PLOT (PHÂN PHỐI MỨC ĐỘ CẢM XÚC)
    # ==============================
    st.subheader("📈 Phân phối điểm số cảm xúc (KDE Plot)")
    try:
        plot_sentiment_kde(df)
    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ KDE: {e}")

    # ==============================
    # 6️⃣ GHI CHÚ DIỄN GIẢI
    # ==============================
    st.markdown("---")
    st.markdown(
        """
        <div style='color:#64748b; font-size:14px;'>
        🔍 <b>Diễn giải:</b><br>
        - Biểu đồ thanh cho thấy tỉ lệ tin tức tích cực, tiêu cực và trung lập.<br>
        - Đường xu hướng thể hiện biến động cảm xúc theo thời gian.<br>
        - Biểu đồ KDE giúp nhận diện cường độ cảm xúc trong từng nhóm tin tức.<br>
        - Nhận xét nhanh và bảng chi tiết giúp đánh giá xu hướng tổng thể của thị trường.
        </div>
        """,
        unsafe_allow_html=True,
    )