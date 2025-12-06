import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_price_data
from utils.charts import create_advanced_chart
from utils import indicators


# ======================================================
# 📘 OVERVIEW TAB
# ======================================================
def render(ticker: str = None):
    """Hiển thị thông tin tổng quan về cổ phiếu và diễn biến giá."""

    # ==============================
    # ⚙️ Lấy cấu hình từ sidebar
    # ==============================
    data_type = st.session_state.get("data_type", "Content")
    time_period = st.session_state.get("time_period", "Before Scandal")
    
    # Lấy các tùy chọn chỉ số kỹ thuật từ session_state
    chart_type = st.session_state.get("chart_type", "Candle")
    show_volume = st.session_state.get("show_volume", True)
    show_sma20 = st.session_state.get("show_sma20", True)
    show_sma50 = st.session_state.get("show_sma50", True)
    show_ema12 = st.session_state.get("show_ema12", False)
    show_ema26 = st.session_state.get("show_ema26", False)
    show_rsi = st.session_state.get("show_rsi", False)
    show_macd = st.session_state.get("show_macd", False)
    show_stoch = st.session_state.get("show_stoch", False)
    show_bb = st.session_state.get("show_bb", False)
    show_adx = st.session_state.get("show_adx", False)
    show_atr = st.session_state.get("show_atr", False)
    show_obv = st.session_state.get("show_obv", False)
    show_vwap = st.session_state.get("show_vwap", False)
    show_fibonacci = st.session_state.get("show_fibonacci", False)

    # ==============================
    # 🧭 Tiêu đề
    # ==============================
    st.markdown(
        f"""
        <h3 style='color:#22c55e'>📘 Tổng quan cổ phiếu {ticker}</h3>
        <p style='color:#94a3b8'>
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ==============================
    # 📑 TAB NAVIGATION
    # ==============================
    tab1, tab2 = st.tabs(["📈 Biểu đồ giá & Kỹ thuật", "💰 Thông tin  tài chính"])
    
    # ==============================
    # TAB 1: BIỂU ĐỒ GIÁ
    # ==============================
    with tab1:
        # ==============================
        # 1️⃣ Thông tin cơ bản doanh nghiệp
        # ==============================
        st.subheader("🏢 Thông tin doanh nghiệp")

        # Lấy thông tin từ VNDirect API
        from utils.vndirect_api import get_vndirect_api
        
        @st.cache_data(ttl=7200, show_spinner=False)
        def get_cached_company_info(ticker_symbol):
            api = get_vndirect_api()
            return api.get_company_info(ticker_symbol)
        
        company_data = get_cached_company_info(ticker)
        
        # Thông tin chi tiết fallback (mở rộng)
        company_details = {
            "FLC": {
                "name": "Công ty Cổ phần Tập đoàn FLC",
                "name_eng": "FLC Group Joint Stock Company",
                "exchange": "HOSE",
                "industry": "Bất động sản",
                "sector": "Tài chính & BĐS",
                "description": "Hoạt động trong lĩnh vực bất động sản, du lịch và hàng không."
            },
            "GAB": {
                "name": "Công ty Cổ phần Đầu tư Khai khoáng & Quản lý tài sản FLC",
                "name_eng": "FLC Stone Joint Stock Company",
                "exchange": "HOSE",
                "industry": "Khai khoáng",
                "sector": "Nguyên vật liệu cơ bản",
                "description": "Công ty thành viên của FLC Group, chuyên khai thác và kinh doanh khoáng sản."
            },
            "HAI": {
                "name": "Công ty Cổ phần Nông dược HAI",
                "name_eng": "HAI Agro Joint Stock Company",
                "exchange": "HOSE",
                "industry": "Nông nghiệp",
                "sector": "Hàng tiêu dùng",
                "description": "Kinh doanh thuốc bảo vệ thực vật và vật tư nông nghiệp."
            },
            "AMD": {
                "name": "Công ty Cổ phần Đầu tư và Khoáng sản FLC",
                "name_eng": "FLC Resources Joint Stock Company",
                "exchange": "HOSE",
                "industry": "Vật liệu xây dựng",
                "sector": "Nguyên vật liệu cơ bản",
                "description": "Chuyên về vật liệu xây dựng và khai khoáng."
            },
            "ART": {
                "name": "Công ty Cổ phần Chứng khoán BOS",
                "name_eng": "BOS Securities Joint Stock Company",
                "exchange": "HOSE",
                "industry": "Chứng khoán",
                "sector": "Tài chính",
                "description": "Cung cấp dịch vụ chứng khoán và đầu tư tài chính."
            },
            "VCB": {
                "name": "Ngân hàng TMCP Ngoại thương Việt Nam",
                "name_eng": "Joint Stock Commercial Bank for Foreign Trade of Vietnam",
                "exchange": "HOSE",
                "industry": "Ngân hàng",
                "sector": "Tài chính",
                "description": "Ngân hàng thương mại cổ phần hàng đầu Việt Nam về vốn điều lệ và quy mô tài sản."
            },
            "CTG": {
                "name": "Ngân hàng TMCP Công thương Việt Nam",
                "name_eng": "Vietnam Joint Stock Commercial Bank for Industry and Trade",
                "exchange": "HOSE",
                "industry": "Ngân hàng",
                "sector": "Tài chính",
                "description": "Ngân hàng thương mại cổ phần lớn nhất Việt Nam theo quy mô mạng lưới."
            },
            "BID": {
                "name": "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam",
                "name_eng": "Joint Stock Commercial Bank for Investment and Development of Vietnam",
                "exchange": "HOSE",
                "industry": "Ngân hàng",
                "sector": "Tài chính",
                "description": "Ngân hàng thương mại cổ phần lớn thứ hai Việt Nam."
            },
        }
        
        if company_data:
            # Hiển thị thông tin công ty từ API
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {company_data.get('company_name', ticker)}")
                if company_data.get('company_name_eng'):
                    st.caption(f"*{company_data.get('company_name_eng')}*")
                
                info_text = f"""
                **Mã CK:** {company_data.get('symbol', 'N/A')}  
                **Sàn giao dịch:** {company_data.get('exchange', 'N/A')}  
                **Ngành:** {company_data.get('industry', 'N/A')}  
                **Nhóm ngành:** {company_data.get('sector', 'N/A')}
                """
                st.info(info_text)
            
            with col2:
                st.markdown("#### 📅 Thông tin khác")
                if company_data.get('established_date'):
                    st.text(f"Ðành lập: {company_data.get('established_date')}")
                if company_data.get('listed_date'):
                    st.text(f"Niêm yết: {company_data.get('listed_date')}")
                if company_data.get('website'):
                    st.markdown(f"[🌐 Website]({company_data.get('website')})")
        else:
            # Fallback: sử dụng thông tin chi tiết từ database nội bộ
            if ticker in company_details:
                details = company_details[ticker]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {details['name']}")
                    st.caption(f"*{details['name_eng']}*")
                    
                    info_text = f"""
                    **Mã CK:** {ticker}  
                    **Sàn giao dịch:** {details['exchange']}  
                    **Ngành:** {details['industry']}  
                    **Nhóm ngành:** {details['sector']}
                    """
                    st.info(info_text)
                    
                    st.markdown(f"📝 **Mô tả:** {details['description']}")
                
                with col2:
                    st.markdown("#### ℹ️ Ghi chú")
                    st.caption("🔄 Dữ liệu nội bộ")
                    st.caption("🌐 API tạm thời không khả dụng")
            else:
                st.warning(f"⚠️ Chưa có thông tin chi tiết cho mã `{ticker}`. API tạm thời không khả dụng.")
                st.caption("💡 Hệ thống vẫn hoạt động bình thường, dữ liệu giá và phân tích không bị ảnh hưởng.")
    
        # ==============================
        # 2️⃣ Biểu đồ giá cổ phiếu
        # ==============================
        st.subheader("💹 Diễn biến giá cổ phiếu")
    
        df_price = load_price_data(ticker)
    
        if df_price.empty:
            st.warning("⚠️ Chưa có dữ liệu giá cổ phiếu để hiển thị.")
            return
    
        # Làm phẳng MultiIndex (nếu có)
        df_price.columns = [c[0] if isinstance(c, tuple) else c for c in df_price.columns]
    
        # Chuẩn hóa tên cột
        df_price.columns = df_price.columns.str.capitalize()
        
        # Đảm bảo có các cột cần thiết: Open, High, Low, Close, Volume
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df_price.columns]
        
        if missing_cols:
            st.error(f"❌ Thiếu các cột: {', '.join(missing_cols)}")
            return
        
        # Ép kiểu dữ liệu về số
        for col in required_cols:
            df_price[col] = pd.to_numeric(df_price[col], errors="coerce")
        
        # FIXED: Lọc dữ liệu nghiêm ngặt hơn TRƯỚC KHI tính các chỉ số
        # Loại bỏ tất cả các hàng có NaN hoặc giá trị <= 0 trong OHLC
        df_price = df_price.dropna(subset=["Open", "High", "Low", "Close"])
        df_price = df_price[(df_price['Open'] > 0) & 
                            (df_price['High'] > 0) & 
                            (df_price['Low'] > 0) & 
                            (df_price['Close'] > 0)]
        
        # ==============================
        # 🔢 Tính toán các chỉ số kỹ thuật
        # ==============================
        selected_indicators = []
        
        # Moving Averages
        if show_sma20:
            df_price = indicators.add_sma(df_price, window=20, name="SMA_20")
            selected_indicators.append("SMA_20")
        
        if show_sma50:
            df_price = indicators.add_sma(df_price, window=50, name="SMA_50")
            selected_indicators.append("SMA_50")
        
        if show_ema12:
            df_price = indicators.add_ema(df_price, span=12, name="EMA_12")
            selected_indicators.append("EMA_12")
        
        if show_ema26:
            df_price = indicators.add_ema(df_price, span=26, name="EMA_26")
            selected_indicators.append("EMA_26")
        
        # Oscillators
        if show_rsi:
            df_price = indicators.add_rsi(df_price)
            selected_indicators.append("RSI")
        
        if show_macd:
            df_price = indicators.add_macd(df_price)
            selected_indicators.append("MACD")
        
        if show_stoch:
            df_price = indicators.add_stoch(df_price)
            selected_indicators.append("Stochastic")
        
        # Trend & Volatility
        if show_bb:
            df_price = indicators.add_bollinger_bands(df_price)
            selected_indicators.append("Bollinger_Bands")
        
        if show_adx:
            df_price = indicators.add_adx(df_price)
            selected_indicators.append("ADX")
        
        if show_atr:
            df_price = indicators.add_atr(df_price)
        
        # Volume Indicators
        if show_obv:
            df_price = indicators.add_obv(df_price)
        
        if show_vwap:
            df_price = indicators.add_vwap(df_price)
            selected_indicators.append("VWAP")
        
        # Tính toán Fibonacci Retracement levels
        fib_levels = {}
        if show_fibonacci:
            fib_levels = indicators.add_fibonacci_levels(df_price, lookback_period=len(df_price))
        
        # ==============================
        # 🎨 Vẽ biểu đồ chuyên nghiệp
        # ==============================
        try:
            fig = create_advanced_chart(
                data=df_price,
                chart_type=chart_type,
                indicators=selected_indicators,
                levels=fib_levels if show_fibonacci else None,
                title=f"📈 {ticker}",
                height=850,  # Tăng chiều cao
                show_volume=show_volume,
                default_visible_days=60  # Hiển thị 60 ngày (2 tháng) để nến to rõ hơn
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Lỗi khi vẽ biểu đồ: {e}")
        
        # ==============================
        # 📋 Bảng tóm tắt chỉ số
        # ==============================
        if selected_indicators:
            st.subheader("📋 Tóm tắt chỉ số kỹ thuật")
            
            indicator_summary = indicators.get_indicator_summary(df_price)
            
            if indicator_summary:
                cols = st.columns(len(indicator_summary))
                
                for idx, (name, values) in enumerate(indicator_summary.items()):
                    with cols[idx]:
                        st.markdown(f"**{name.replace('_', ' ')}**")
                        for key, val in values.items():
                            # Tô màu tín hiệu
                            if key == "Tín hiệu":
                                if "BUY" in str(val):
                                    st.success(f"🟢 {val}")
                                elif "SELL" in str(val):
                                    st.error(f"🔴 {val}")
                                else:
                                    st.info(f"🟡 {val}")
                            else:
                                st.text(f"{key}: {val}")
            else:
                st.info("Chọn các chỉ số từ sidebar để xem tóm tắt.")
        
        # ==============================
        # 📊 Hiển thị Fibonacci Retracement Levels
        # ==============================
        if show_fibonacci and fib_levels:
            st.subheader("📊 Fibonacci Retracement Levels")
            
            st.markdown("""
            **Fibonacci Retracement** là công cụ phân tích kỹ thuật dựa trên dãy số Fibonacci để xác định 
            các mức hỗ trợ và kháng cự tiềm năng. Các mức quan trọng:
            - **61.8% (Golden Ratio)**: Mức thoái lui quan trọng nhất
            - **50%**: Mức tâm lý quan trọng
            - **38.2%** và **23.6%**: Mức hỗ trợ/kháng cự phụ
            """)
            
            # Tạo bảng hiển thị các mức Fibonacci
            fib_df = pd.DataFrame([
                {"Mức": k, "Giá": f"{v:,.0f} VNĐ"} 
                for k, v in fib_levels.items()
            ])
            
            # Highlight mức quan trọng
            def highlight_important(row):
                if "50%" in row["Mức"] or "61.8%" in row["Mức"]:
                    return ['background-color: #FFD54F; color: black'] * len(row)
                elif "0%" in row["Mức"] or "100%" in row["Mức"]:
                    return ['background-color: #64B5F6; color: white'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                fib_df.style.apply(highlight_important, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            # Giá hiện tại so với Fibonacci
            current_price = df_price['Close'].iloc[-1]
            st.metric(
                label="Giá hiện tại",
                value=f"{current_price:,.0f} VNĐ",
                delta=None
            )
    
            # ==============================
            # 3️⃣ Ghi chú giai đoạn scandal
            # ==============================
            st.markdown("---")
            if time_period == "Before Scandal":
                st.info(
                    f"""
                    🕓 **Giai đoạn Trước Scandal:**  
                    Dữ liệu phản ánh tin tức và tâm lý thị trường **trước khi sự kiện tiêu cực liên quan đến {ticker} xảy ra.**  
                    Mục tiêu là đánh giá trạng thái tâm lý ổn định và xu hướng giá bình thường của nhà đầu tư.
                    """
                )
            else:
                st.warning(
                    f"""
                    ⚠️ **Giai đoạn Sau Scandal:**  
                    Dữ liệu phản ánh cảm xúc và phản ứng của thị trường **sau khi các bê bối hoặc tin tức tiêu cực được công bố.**  
                    Giai đoạn này thường cho thấy sự sụt giảm niềm tin và biến động giá mạnh.
                    """
            )
    
    # ==============================
    # TAB 2: THÔNG TIN TÀI CHÍNH
    # ==============================
    with tab2:
        st.subheader("💰 Thông tin tài chính")
        
        # Tabs con cho các báo cáo tài chính
        fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs([
            "📊 Bảng cân đối kế toán", 
            "💵 Báo cáo kết quả kinh doanh",
            "💸 Báo cáo lưu chuyển tiền tệ",
            "📈 Chỉ số tài chính"
        ])
        
        # Balance Sheet
        with fin_tab1:
            st.markdown("### 📊 Bảng cân đối kế toán (Balance Sheet)")
            
            try:
                import warnings
                warnings.filterwarnings('ignore')
                from vnstock import Vnstock
                stock = Vnstock().stock(symbol=ticker, source='VCI')
                
                # Lấy balance sheet
                balance_sheet = stock.finance.balance_sheet(period='quarter', lang='vi')
                
                if balance_sheet is not None and not balance_sheet.empty:
                    # Lấy dữ liệu quý gần nhất (cột đầu tiên)
                    latest_quarter = balance_sheet.columns[0]
                    
                    # Convert index sang string để tránh lỗi .str accessor
                    balance_sheet.index = balance_sheet.index.astype(str)
                    
                    # Hiển thị bảng đầy đủ
                    st.dataframe(balance_sheet, use_container_width=True)
                else:
                    st.warning(f"⚠️ Không có dữ liệu bảng cân đối kế toán cho {ticker}")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
                st.info("💡 Thử chọn mã cổ phiếu khác hoặc kiểm tra kết nối mạng")
        
        # Income Statement
        with fin_tab2:
            st.markdown("### 💵 Báo cáo kết quả kinh doanh (Income Statement)")
            
            try:
                import warnings
                warnings.filterwarnings('ignore')
                from vnstock import Vnstock
                stock = Vnstock().stock(symbol=ticker, source='VCI')
                
                # Lấy income statement
                income_statement = stock.finance.income_statement(period='quarter', lang='vi')
                
                if income_statement is not None and not income_statement.empty:
                    latest_quarter = income_statement.columns[0]
                    
                    # Convert index sang string
                    income_statement.index = income_statement.index.astype(str)                   
                    # Hiển thị bảng đầy đủ
                    st.dataframe(income_statement, use_container_width=True)
                else:
                    st.warning(f"⚠️ Không có dữ liệu báo cáo kết quả kinh doanh cho {ticker}")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
                st.info("💡 Thử chọn mã cổ phiếu khác hoặc kiểm tra kết nối mạng")
        
        # Cash Flow Statement
        with fin_tab3:
            st.markdown("### 💸 Báo cáo lưu chuyển tiền tệ (Cash Flow Statement)")
            
            try:
                import warnings
                warnings.filterwarnings('ignore')
                from vnstock import Vnstock
                stock = Vnstock().stock(symbol=ticker, source='VCI')
                
                # Lấy cash flow statement
                cash_flow = stock.finance.cash_flow(period='quarter', lang='vi')
                
                if cash_flow is not None and not cash_flow.empty:
                    latest_quarter = cash_flow.columns[0]
                    
                    # Convert index sang string
                    cash_flow.index = cash_flow.index.astype(str)
                    # Hiển thị bảng đầy đủ
                    st.dataframe(cash_flow, use_container_width=True)
                else:
                    st.warning(f"⚠️ Không có dữ liệu báo cáo lưu chuyển tiền tệ cho {ticker}")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
                st.info("💡 Thử chọn mã cổ phiếu khác hoặc kiểm tra kết nối mạng")
        
        # Financial Ratios
        with fin_tab4:
            st.markdown("### 📈 Chỉ số tài chính (Financial Ratios)")
            
            try:
                import warnings
                warnings.filterwarnings('ignore')
                from vnstock import Vnstock
                stock = Vnstock().stock(symbol=ticker, source='VCI')
                
                # Lấy financial ratios
                ratios = stock.finance.ratio(period='quarter', lang='vi')
                
                if ratios is not None and not ratios.empty:
                    latest_quarter = ratios.columns[0]
                    
                    # Convert index sang string
                    ratios.index = ratios.index.astype(str)
                    # Hiển thị bảng đầy đủ
                    st.dataframe(ratios, use_container_width=True)
                else:
                    st.warning(f"⚠️ Không có dữ liệu chỉ số tài chính cho {ticker}")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
                st.info("💡 Thử chọn mã cổ phiếu khác hoặc kiểm tra kết nối mạng")
        
        st.markdown("---")
        st.caption("💡 **Lưu ý:** Các báo cáo tài chính sẽ được cập nhật realtime từ VNDirect API hoặc Vnstock trong phiên bản tiếp theo.")
