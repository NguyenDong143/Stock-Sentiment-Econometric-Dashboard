# ======================================================
# 📊 ui/granger_tab.py — Kiểm định Granger (Cải tiến)
# ======================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import grangercausalitytests

# ✅ Import module nội bộ
from utils.data_loader import load_granger_data
from models.granger_test import granger_test  # VAR-based nâng cao


# ======================================================
# 📘 KẾT QUẢ NGHIÊN CỨU CHÍNH THỨC (Tables IV, V trong paper)
# ======================================================
GRANGER_RESULTS = {
    "Before Scandal": {
        "AMD": {"Lag": 10, "Coefficient": 69.65, "p-value": 0.015, "Kết luận": "✅ Có quan hệ nhân quả"},
        "ART": {"Lag": "-", "Coefficient": "-", "p-value": "-", "Kết luận": "❌ Không có quan hệ"},
        "FLC": {"Lag": 10, "Coefficient": 110.68, "p-value": 0.031, "Kết luận": "✅ Có quan hệ nhân quả"},
        "GAB": {"Lag": "-", "Coefficient": "-", "p-value": "-", "Kết luận": "❌ Không có quan hệ"},
        "HAI": {"Lag": 10, "Coefficient": 53.09, "p-value": 0.040, "Kết luận": "✅ Có quan hệ nhân quả"},
    },
    "After Scandal": {
        "AMD": {"Lag": 5, "Coefficient": -36.65, "p-value": 0.085, "Kết luận": "⚠️ Quan hệ biên (10%)"},
        "ART": {"Lag": 5, "Coefficient": -70.81, "p-value": 0.028, "Kết luận": "✅ Có quan hệ nhân quả"},
        "FLC": {"Lag": 6, "Coefficient": 71.05, "p-value": 0.055, "Kết luận": "⚠️ Quan hệ biên (10%)"},
        "GAB": {"Lag": "-", "Coefficient": "-", "p-value": "-", "Kết luận": "❌ Không có quan hệ"},
        "HAI": {"Lag": 7, "Coefficient": 39.36, "p-value": 0.077, "Kết luận": "⚠️ Quan hệ biên (10%)"},
    },
}


# ======================================================
# 🎨 HÀM PHỤ TRỢ
# ======================================================
def format_pvalue(pval):
    """Format p-value với dấu sao ý nghĩa thống kê"""
    try:
        pval = float(pval)
        if pval < 0.01:
            return f"{pval:.4f}***"
        elif pval < 0.05:
            return f"{pval:.4f}**"
        elif pval < 0.1:
            return f"{pval:.4f}*"
        else:
            return f"{pval:.4f}"
    except:
        return str(pval)


def create_granger_heatmap(results_df):
    """Tạo heatmap cho kết quả Granger causality"""
    if results_df.empty or 'Biến gây ảnh hưởng' not in results_df.columns:
        return None
    
    # Tạo ma trận p-value
    pivot_data = []
    
    for _, row in results_df.iterrows():
        causing = row['Biến gây ảnh hưởng']
        caused = row['Biến bị ảnh hưởng']
        pval = row['p-value']
        
        pivot_data.append({
            'from': causing,
            'to': caused,
            'p_value': pval,
            'significant': '✅' if pval < 0.05 else '❌'
        })
    
    if not pivot_data:
        return None
    
    df_plot = pd.DataFrame(pivot_data)
    
    # Tạo heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df_plot['p_value'],
        x=df_plot['to'],
        y=df_plot['from'],
        colorscale='RdYlGn_r',
        text=df_plot['significant'],
        texttemplate='%{text}',
        colorbar=dict(title="p-value"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Ma trận quan hệ nhân quả Granger",
        xaxis_title="Biến bị ảnh hưởng",
        yaxis_title="Biến gây ảnh hưởng",
        height=400
    )
    
    return fig


# ======================================================
# 🧠 TAB KIỂM ĐỊNH GRANGER
# ======================================================
def render(ticker: str = None):
    st.header("🔁 Kiểm định Nhân quả Granger")
    
    st.markdown("""
    """, unsafe_allow_html=True)

    # --- Thông tin cấu hình ---
    data_type = st.session_state.get("data_type", "Content")
    time_period = st.session_state.get("time_period", "Before Scandal")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Loại dữ liệu", data_type)
    with col2:
        st.metric("📅 Giai đoạn", time_period)
    with col3:
        st.metric("📈 Mã CK", ticker if ticker else "Chưa chọn")

    st.divider()

    # ======================================================
    # 📊 KẾT QUẢ NGHIÊN CỨU CHÍNH THỨC
    # ======================================================
    if ticker and ticker in GRANGER_RESULTS.get(time_period, {}):
        st.subheader("📘 Kết quả nghiên cứu (2018–2023)")
        
        res = GRANGER_RESULTS[time_period][ticker]
        
        # Hiển thị dạng bảng đẹp hơn
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lag tối ưu", res["Lag"])
        with col2:
            st.metric("Coefficient", res["Coefficient"] if res["Coefficient"] != "-" else "N/A")
        with col3:
            st.metric("P-value", format_pvalue(res["p-value"]) if res["p-value"] != "-" else "N/A")
        with col4:
            st.info(res["Kết luận"])
        
        # Giải thích kết quả
        pval = res["p-value"]
        if isinstance(pval, (int, float)):
            if pval < 0.05:
                st.success(
                    f"✅ **Kết luận:** Tin tức có tác động nhân quả đến giá cổ phiếu **{ticker}** "
                    f"với mức ý nghĩa 5% (p = {pval:.4f})"
                )
            elif pval < 0.1:
                st.info(
                    f"⚠️ **Kết luận:** Quan hệ nhân quả tồn tại ở mức biên 10% (p = {pval:.4f}). "
                    f"Tác động yếu hơn so với mức ý nghĩa tiêu chuẩn."
                )
            else:
                st.warning(
                    f"❌ **Kết luận:** Không phát hiện quan hệ nhân quả có ý nghĩa thống kê "
                    f"(p = {pval:.4f} ≥ 0.05)"
                )
        else:
            st.warning("❌ Không có dữ liệu đủ để kiểm định trong giai đoạn này.")
        
        st.divider()

    # ======================================================
    # ⚙️ KIỂM ĐỊNH TƯƠNG TÁC
    # ======================================================
    st.subheader("⚙️ Thực hiện kiểm định Granger tương tác")
    
    # Chọn loại kiểm định
    test_mode = st.radio(
        "**Chọn phương pháp kiểm định:**",
        [
            "🔹 Kiểm định Granger đơn biến (Classic)",
            "🔸 Kiểm định VAR-based đa biến (Nâng cao - theo Paper)"
        ],
        index=1,
        help="VAR-based cho phép kiểm tra nhiều biến cùng lúc và xử lý chuỗi không dừng tự động"
    )

    # Load dữ liệu với cache (đã có @st.cache_data trong load_granger_data)
    with st.spinner("Đang tải dữ liệu..."):
        df = load_granger_data(ticker, data_type, time_period)
    
    if df is None or df.empty:
        st.warning("⚠️ Không tìm thấy dữ liệu để kiểm định. Vui lòng chọn mã cổ phiếu và kiểm tra dữ liệu.")
        return

    # Lọc các cột số
    available_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if len(available_cols) < 2:
        st.error("❌ Dữ liệu cần ít nhất 2 biến số (ví dụ: sentiment_score, stock_price).")
        st.info("💡 Các cột hiện có: " + ", ".join(df.columns.tolist()))
        return

    st.info(f"📊 Dữ liệu: {len(df)} quan sát | {len(available_cols)} biến số")

    # ======================================================
    # 🧮 CLASSIC GRANGER TEST (Đơn biến)
    # ======================================================
    if "Classic" in test_mode:
        st.markdown("### 🔧 Cấu hình kiểm định đơn biến")
        
        col1, col2 = st.columns(2)
        
        with col1:
            y_col = st.selectbox(
                "🔹 Biến bị tác động (Dependent Variable)",
                available_cols,
                index=0,
                help="Thường là giá cổ phiếu (stock_price)"
            )
        
        with col2:
            x_col = st.selectbox(
                "🔹 Biến gây tác động (Independent Variable)",
                available_cols,
                index=1 if len(available_cols) > 1 else 0,
                help="Thường là điểm sentiment (sentiment_score)"
            )
        
        maxlag = st.slider(
            "⏱ Độ trễ tối đa (lag)",
            min_value=1,
            max_value=10,
            value=5,
            help="Số ngày quá khứ tối đa để kiểm tra ảnh hưởng. Giảm lag để tăng tốc."
        )

        if y_col == x_col:
            st.warning("⚠️ Hai biến phải khác nhau để kiểm định nhân quả.")
            return

        if st.button("🚀 Chạy kiểm định Classic Granger", type="primary", use_container_width=True):
            df_test = df[[y_col, x_col]].dropna().copy()
            df_test.columns = ["y", "x"]
            
            if len(df_test) < maxlag + 10:
                st.error(f"❌ Không đủ dữ liệu: cần ít nhất {maxlag + 10} quan sát, hiện có {len(df_test)}")
                return

            @st.cache_data(show_spinner=False, ttl=7200)
            def run_granger_test_cached(test_data_tuple, max_lag):
                # Convert tuple back to DataFrame
                import pandas as pd
                df_temp = pd.DataFrame(test_data_tuple, columns=['y', 'x'])
                return grangercausalitytests(df_temp, maxlag=max_lag, verbose=False)
            
            with st.spinner(f"🔍 Đang chạy kiểm định Granger (lag ≤ {maxlag})..."):
                try:
                    # Convert to tuple for caching
                    test_data_tuple = tuple(map(tuple, df_test.values))
                    results = run_granger_test_cached(test_data_tuple, maxlag)
                    
                    # Trích xuất p-values
                    pvals = []
                    fstats = []
                    
                    for i in range(maxlag):
                        if (i + 1) in results and "ssr_ftest" in results[i + 1][0]:
                            fstats.append(round(results[i + 1][0]["ssr_ftest"][0], 4))
                            pvals.append(round(results[i + 1][0]["ssr_ftest"][1], 4))
                        else:
                            fstats.append(None)
                            pvals.append(None)
                    
                    df_result = pd.DataFrame({
                        "Lag": range(1, maxlag + 1),
                        "F-statistic": fstats,
                        "p-value": pvals,
                        "Ý nghĩa": ["✅" if p and p < 0.05 else "❌" if p else "N/A" for p in pvals]
                    })
                    
                    st.subheader("📊 Kết quả kiểm định")
                    st.dataframe(
                        df_result.style.format({
                            "F-statistic": "{:.4f}",
                            "p-value": "{:.4f}"
                        }).applymap(
                            lambda x: 'background-color: #d1fae5' if x == "✅" else '',
                            subset=['Ý nghĩa']
                        ),
                        use_container_width=True
                    )

                    # Vẽ biểu đồ p-value
                    fig = px.line(
                        df_result,
                        x="Lag",
                        y="p-value",
                        markers=True,
                        title=f"📈 Granger Causality: {x_col} → {y_col}",
                        labels={"p-value": "P-value", "Lag": "Độ trễ (ngày)"}
                    )
                    fig.add_hline(
                        y=0.05,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Ngưỡng α = 0.05",
                        annotation_position="right"
                    )
                    fig.add_hline(
                        y=0.1,
                        line_dash="dot",
                        line_color="orange",
                        annotation_text="Ngưỡng α = 0.10",
                        annotation_position="right"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Kết luận
                    sig_lags = [lag for lag, p in zip(range(1, maxlag + 1), pvals) if p and p < 0.05]
                    if sig_lags:
                        st.success(
                            f"✅ **Kết luận:** Tồn tại quan hệ nhân quả **{x_col} → {y_col}** "
                            f"tại độ trễ: **{sig_lags}** (p < 0.05)"
                        )
                    else:
                        st.info(
                            f"❌ **Kết luận:** Không phát hiện quan hệ nhân quả có ý nghĩa thống kê "
                            f"**{x_col} → {y_col}** (tất cả p-value ≥ 0.05)"
                        )

                except Exception as e:
                    st.error(f"❌ Lỗi khi chạy kiểm định Granger: {str(e)}")
                    with st.expander("🔍 Chi tiết lỗi"):
                        st.code(str(e))

    # ======================================================
    # 🧠 VAR-BASED GRANGER TEST (Đa biến - Theo Paper)
    # ======================================================
    else:
        st.markdown("### 🔧 Cấu hình kiểm định VAR-based (theo Paper)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cols_selected = st.multiselect(
                "**Chọn các biến để phân tích:**",
                options=available_cols,
                default=available_cols[:min(3, len(available_cols))],
                help="Chọn ít nhất 2 biến. Thường bao gồm sentiment_score và stock_price"
            )
        
        with col2:
            maxlag = st.slider(
                "⏱ **Độ trễ tối đa**",
                min_value=1,
                max_value=14,
                value=10,
                help="Theo paper: lag = 10 cho giai đoạn Before, 5-7 cho After"
            )
        
        # Tùy chọn nâng cao
        with st.expander("⚙️ Tùy chọn nâng cao"):
            test_individually = st.checkbox(
                "Test từng biến riêng lẻ (pairwise)",
                value=False,
                help="Nếu chọn, sẽ test từng cặp biến riêng biệt như trong Tables IV, V của paper"
            )
            significance_level = st.select_slider(
                "Mức ý nghĩa thống kê (α)",
                options=[0.01, 0.05, 0.1],
                value=0.05
            )

        if len(cols_selected) < 2:
            st.warning("⚠️ Cần chọn ít nhất 2 biến để thực hiện kiểm định VAR-based.")
            return

        if st.button("🚀 Chạy kiểm định VAR-based Granger", type="primary", use_container_width=True):
            with st.spinner("🧮 Đang chạy kiểm định VAR-based Granger..."):
                try:
                    @st.cache_data(show_spinner=False, ttl=7200)
                    def run_var_granger_cached(df_data, cols, maxlag, test_indiv, sig_level):
                        return granger_test(
                            df=df_data,
                            columns_to_test=cols,
                            maxlags=maxlag,
                            test_individually=test_indiv,
                            significance_level=sig_level
                        )
                    
                    # Gọi hàm granger_test từ models
                    results_df, var_model = run_var_granger_cached(
                        df,
                        cols_selected,
                        maxlag,
                        test_individually,
                        significance_level
                    )
                    
                    if results_df is None or results_df.empty:
                        st.warning("⚠️ Không có kết quả hợp lệ. Vui lòng kiểm tra dữ liệu hoặc giảm số lag.")
                        return

                    # Hiển thị kết quả
                    st.subheader("📈 Kết quả VAR-based Granger Test")
                    
                    # Style DataFrame
                    styled_df = results_df.style.format({
                        "Coef (TB)": "{:.6f}",
                        "F-statistic": "{:.4f}",
                        "p-value": "{:.4f}"
                    }).applymap(
                        lambda x: 'background-color: #d1fae5' if x == "✅" else 'background-color: #fee2e2' if x == "❌" else '',
                        subset=['Có ý nghĩa'] if 'Có ý nghĩa' in results_df.columns else []
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)

                    # Tóm tắt kết quả
                    sig_rows = results_df[results_df["p-value"] < significance_level]
                    total_tests = len(results_df)
                    sig_tests = len(sig_rows)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tổng số test", total_tests)
                    with col2:
                        st.metric("Có ý nghĩa", sig_tests)
                    with col3:
                        st.metric("Tỷ lệ", f"{sig_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%")

                    # Heatmap (nếu test individually)
                    if test_individually and not results_df.empty:
                        fig_heatmap = create_granger_heatmap(results_df)
                        if fig_heatmap:
                            st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Kết luận chi tiết
                    if not sig_rows.empty:
                        st.success(f"✅ Phát hiện {sig_tests} quan hệ nhân quả có ý nghĩa thống kê:")
                        
                        for idx, row in sig_rows.iterrows():
                            cause = row["Biến gây ảnh hưởng"]
                            effect = row["Biến bị ảnh hưởng"]
                            pval = row["p-value"]
                            fstat = row["F-statistic"]
                            coef = row.get("Coef (TB)", "N/A")
                            
                            st.markdown(
                                f"- **{cause}** → **{effect}**: "
                                f"F = {fstat:.2f}, p = {format_pvalue(pval)}, Coef = {coef:.6f}"
                            )
                    else:
                        st.info(
                            f"❌ Không phát hiện mối quan hệ nhân quả có ý nghĩa thống kê "
                            f"(tất cả p-value ≥ {significance_level})"
                        )
                    
                    # Thông tin mô hình VAR
                    if var_model:
                        with st.expander("📊 Thông tin mô hình VAR"):
                            st.write(f"**Số phương trình:** {var_model.neqs}")
                            st.write(f"**Số quan sát:** {var_model.nobs}")
                            st.write(f"**Lag sử dụng:** {var_model.k_ar}")
                            
                            if hasattr(var_model, 'is_stable'):
                                is_stable = var_model.is_stable()
                                if is_stable:
                                    st.success("✅ Mô hình VAR ổn định")
                                else:
                                    st.warning("⚠️ Mô hình VAR không ổn định")

                except Exception as e:
                    st.error(f"❌ Lỗi khi chạy kiểm định VAR-based: {str(e)}")
                    with st.expander("🔍 Chi tiết lỗi"):
                        st.code(str(e))

    # ======================================================
    # 📚 HƯỚNG DẪN & GHI CHÚ
    # ======================================================
    st.divider()
    with st.expander("💡 Mẹo sử dụng"):
        st.markdown("""
        1. **Chọn giai đoạn phù hợp:** Before Scandal thường có quan hệ nhân quả mạnh hơn
        2. **Lag phù hợp:** 
           - Before Scandal: thử lag = 10
           - After Scandal: thử lag = 5-7
        3. **Kiểm tra dữ liệu:** Cần ít nhất 50-100 quan sát để kết quả đáng tin cậy
        4. **So sánh với Paper:** Đối chiếu kết quả với Tables IV, V trong nghiên cứu gốc
        """)