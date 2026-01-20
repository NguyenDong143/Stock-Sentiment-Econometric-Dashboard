import streamlit as st
import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# ======================================================
# ☁️ WORD CLOUD TAB
# ======================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "data_world_cloud")

# Vietnamese stopwords
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'là', 'có', 'được', 'trong', 'cho', 'các', 'với', 'này',
    'để', 'đến', 'người', 'những', 'không', 'một', 'như', 'khi', 'từ', 'năm',
    'theo', 'đã', 'về', 'cũng', 'nhưng', 'tại', 'hay', 'sẽ', 'còn', 'ra',
    'nhiều', 'đang', 'hơn', 'đó', 'sau', 'rất', 'vào', 'lại', 'thì', 'nên',
    'trên', 'mà', 'đi', 'do', 'bị', 'phải', 'chỉ', 'họ', 'nếu', 'tuy',
    'vì', 'bằng', 'trước', 'ở', 'lên', 'việc', 'hoặc', 'nào', 'dù', 'thế',
    'rằng', 'bởi', 'ai', 'nói', 'làm', 'thêm', 'qua', 'giữa', 'đây', 'tới',
    'số', 'đều', 'vẫn', 'chưa', 'ngày', 'hiện', 'gì', 'thì', 'mới', 'luôn',
    'the', 'and', 'of', 'to', 'in', 'for', 'is', 'on', 'that', 'with'
}


def load_wordcloud_data(year: str) -> pd.DataFrame:
    """Load data for word cloud from Excel file"""
    try:
        file_path = os.path.join(DATA_DIR, f"cleaned_data_vneconomy_{year}.xlsx")
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            return df
        else:
            st.error(f"❌ File không tồn tại: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {e}")
        return pd.DataFrame()


def get_available_years():
    """Get list of available years from data files"""
    years = []
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.startswith("cleaned_data_vneconomy_") and file.endswith(".xlsx"):
                year = file.replace("cleaned_data_vneconomy_", "").replace(".xlsx", "")
                years.append(year)
    return sorted(years)


def preprocess_text(text: str) -> str:
    """Clean and preprocess text for word cloud"""
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_word_frequencies(texts: list) -> dict:
    """Calculate word frequencies from list of texts"""
    all_words = []
    for text in texts:
        cleaned = preprocess_text(text)
        words = cleaned.split()
        # Filter stopwords and short words
        words = [w for w in words if w not in VIETNAMESE_STOPWORDS and len(w) > 2]
        all_words.extend(words)
    
    return dict(Counter(all_words))


def create_wordcloud(word_freq: dict, colormap: str = 'viridis') -> WordCloud:
    """Generate word cloud from word frequencies"""
    wc = WordCloud(
        width=1200,
        height=600,
        background_color='#0f172a',
        colormap=colormap,
        max_words=200,
        min_font_size=10,
        max_font_size=150,
        random_state=42,
        prefer_horizontal=0.7,
        contour_color='#22c55e',
        contour_width=2,
    )
    
    if word_freq:
        wc.generate_from_frequencies(word_freq)
    
    return wc


def render():
    """
    Tab hiển thị Word Cloud từ dữ liệu tin tức VnEconomy.
    Cho phép chọn năm và loại cột để tạo word cloud.
    """
    
    st.markdown(
        """
        <h3 style='color:#8b5cf6'>☁️ Word Cloud - Phân tích từ khóa tin tức</h3>
        <p style='color:#94a3b8'>
        Trực quan hóa các từ khóa phổ biến trong tin tức kinh tế theo từng năm.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    # ==============================
    # 📅 CHỌN NĂM VÀ CẤU HÌNH
    # ==============================
    available_years = get_available_years()
    
    if not available_years:
        st.warning("⚠️ Không tìm thấy dữ liệu Word Cloud trong thư mục data/data_world_cloud/")
        return
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        selected_year = st.selectbox(
            "📅 Chọn năm:",
            options=available_years,
            index=len(available_years) - 1,  # Default to latest year
            key="wordcloud_year"
        )
    
    with col2:
        colormap_options = {
            "🌈 Viridis": "viridis",
            "🔥 Plasma": "plasma",
            "🌅 Inferno": "inferno",
            "🍃 Greens": "Greens",
            "💎 Blues": "Blues",
            "🌸 Purples": "Purples",
            "☀️ YlOrRd": "YlOrRd",
            "🌊 Ocean": "ocean",
        }
        selected_colormap = st.selectbox(
            "🎨 Bảng màu:",
            options=list(colormap_options.keys()),
            index=0,
            key="wordcloud_colormap"
        )
    
    with col3:
        max_words = st.slider(
            "📝 Số từ tối đa:",
            min_value=50,
            max_value=300,
            value=150,
            step=25,
            key="wordcloud_max_words"
        )
    
    # ==============================
    # 📂 TẢI VÀ XỬ LÝ DỮ LIỆU
    # ==============================
    with st.spinner(f"Đang tải dữ liệu năm {selected_year}..."):
        df = load_wordcloud_data(selected_year)
    
    if df.empty:
        return
    
    # Hiển thị thông tin về dữ liệu
    st.markdown("---")
    
    # Chọn cột để tạo word cloud
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    if not text_columns:
        st.error("❌ Không tìm thấy cột văn bản trong dữ liệu.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_column = st.selectbox(
            "📋 Chọn cột dữ liệu:",
            options=text_columns,
            index=0,
            key="wordcloud_column"
        )
    
    with col2:
        st.metric("📊 Số bài viết", f"{len(df):,}")
    
    # ==============================
    # ☁️ TẠO WORD CLOUD
    # ==============================
    with st.spinner("Đang tạo Word Cloud..."):
        texts = df[selected_column].dropna().tolist()
        word_freq = get_word_frequencies(texts)
        
        if not word_freq:
            st.warning("⚠️ Không có đủ dữ liệu để tạo Word Cloud.")
            return
        
        # Update WordCloud with user settings
        wc = WordCloud(
            width=1200,
            height=600,
            background_color='#0f172a',
            colormap=colormap_options[selected_colormap],
            max_words=max_words,
            min_font_size=10,
            max_font_size=150,
            random_state=42,
            prefer_horizontal=0.7,
            contour_color='#22c55e',
            contour_width=2,
        )
        wc.generate_from_frequencies(word_freq)
    
    # ==============================
    # 📊 HIỂN THỊ KẾT QUẢ
    # ==============================
    st.markdown("---")
    st.subheader(f"☁️ Word Cloud - Tin tức {selected_year}")
    
    # Hiển thị word cloud
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor('#0f172a')
    plt.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)
    
    # ==============================
    # 📈 TOP KEYWORDS
    # ==============================
    st.markdown("---")
    st.subheader("🔤 Top từ khóa phổ biến nhất")
    
    # Sắp xếp và lấy top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tạo bar chart cho top 10
        top_10 = sorted_words[:10]
        words, counts = zip(*top_10) if top_10 else ([], [])
        
        st.markdown("#### 📊 Top 10 Keywords")
        
        # Tạo plotly bar chart
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Bar(
            x=list(counts)[::-1],
            y=list(words)[::-1],
            orientation='h',
            marker=dict(
                color=list(range(10)),
                colorscale='Viridis',
                line=dict(color='rgba(50,50,50,0.8)', width=1)
            ),
            text=list(counts)[::-1],
            textposition='outside',
            textfont=dict(color='#e2e8f0')
        ))
        
        fig.update_layout(
            height=400,
            plot_bgcolor='#0f172a',
            paper_bgcolor='#0f172a',
            font=dict(color='#e2e8f0'),
            xaxis=dict(
                title='Số lần xuất hiện',
                gridcolor='#1e293b',
                showgrid=True
            ),
            yaxis=dict(
                title='',
                showgrid=False
            ),
            margin=dict(l=10, r=20, t=10, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Bảng chi tiết Top 20")
        
        # Tạo DataFrame để hiển thị
        top_df = pd.DataFrame(sorted_words[:20], columns=['Từ khóa', 'Số lần xuất hiện'])
        top_df['Thứ hạng'] = range(1, len(top_df) + 1)
        top_df = top_df[['Thứ hạng', 'Từ khóa', 'Số lần xuất hiện']]
        
        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    # ==============================
    # 📊 THỐNG KÊ TỔNG QUAN
    # ==============================
    st.markdown("---")
    st.subheader("📊 Thống kê tổng quan")
    
    total_words = sum(word_freq.values())
    unique_words = len(word_freq)
    avg_frequency = total_words / unique_words if unique_words > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #22c55e22, #22c55e11); 
                        padding: 20px; border-radius: 15px; text-align: center;
                        border: 1px solid #22c55e44;'>
                <h2 style='color: #22c55e; margin: 0;'>{total_words:,}</h2>
                <p style='color: #94a3b8; margin: 5px 0 0 0;'>📝 Tổng số từ</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #3b82f622, #3b82f611); 
                        padding: 20px; border-radius: 15px; text-align: center;
                        border: 1px solid #3b82f644;'>
                <h2 style='color: #3b82f6; margin: 0;'>{unique_words:,}</h2>
                <p style='color: #94a3b8; margin: 5px 0 0 0;'>🔤 Từ khóa duy nhất</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #8b5cf622, #8b5cf611); 
                        padding: 20px; border-radius: 15px; text-align: center;
                        border: 1px solid #8b5cf644;'>
                <h2 style='color: #8b5cf6; margin: 0;'>{avg_frequency:.1f}</h2>
                <p style='color: #94a3b8; margin: 5px 0 0 0;'>📈 Tần suất TB</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #f59e0b22, #f59e0b11); 
                        padding: 20px; border-radius: 15px; text-align: center;
                        border: 1px solid #f59e0b44;'>
                <h2 style='color: #f59e0b; margin: 0;'>{len(df):,}</h2>
                <p style='color: #94a3b8; margin: 5px 0 0 0;'>📰 Số bài viết</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ==============================
    # 📌 SO SÁNH NHIỀU NĂM
    # ==============================
    st.markdown("---")
    st.subheader("📆 So sánh xu hướng từ khóa qua các năm")
    
    if len(available_years) > 1:
        compare_years = st.multiselect(
            "Chọn các năm để so sánh:",
            options=available_years,
            default=available_years[-2:] if len(available_years) >= 2 else available_years,
            key="compare_years"
        )
        
        if len(compare_years) >= 2:
            comparison_data = {}
            
            for year in compare_years:
                df_year = load_wordcloud_data(year)
                if not df_year.empty and selected_column in df_year.columns:
                    texts_year = df_year[selected_column].dropna().tolist()
                    freq_year = get_word_frequencies(texts_year)
                    # Get top 10 words
                    top_words = sorted(freq_year.items(), key=lambda x: x[1], reverse=True)[:10]
                    comparison_data[year] = dict(top_words)
            
            if comparison_data:
                # Tạo comparison chart
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Lấy tất cả các từ khóa từ tất cả các năm
                all_keywords = set()
                for year_data in comparison_data.values():
                    all_keywords.update(year_data.keys())
                
                # Top 15 keywords dựa trên tổng frequency
                keyword_totals = {}
                for keyword in all_keywords:
                    keyword_totals[keyword] = sum(
                        comparison_data[year].get(keyword, 0) 
                        for year in comparison_data
                    )
                
                top_keywords = sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)[:15]
                top_keyword_names = [k for k, v in top_keywords]
                
                colors = ['#22c55e', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4']
                
                for i, year in enumerate(compare_years):
                    values = [comparison_data[year].get(k, 0) for k in top_keyword_names]
                    fig.add_trace(go.Bar(
                        name=f'Năm {year}',
                        x=top_keyword_names,
                        y=values,
                        marker_color=colors[i % len(colors)]
                    ))
                
                fig.update_layout(
                    barmode='group',
                    height=450,
                    plot_bgcolor='#0f172a',
                    paper_bgcolor='#0f172a',
                    font=dict(color='#e2e8f0'),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    xaxis=dict(
                        title='Từ khóa',
                        tickangle=45,
                        showgrid=False
                    ),
                    yaxis=dict(
                        title='Số lần xuất hiện',
                        gridcolor='#1e293b',
                        showgrid=True
                    ),
                    margin=dict(l=10, r=10, t=40, b=80)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Cần có dữ liệu từ ít nhất 2 năm để so sánh.")
    
    # ==============================
    # 📌 GHI CHÚ
    # ==============================
    st.markdown("---")
    st.markdown(
        """
        <div style='color:#64748b; font-size:14px;'>
        🔍 <b>Diễn giải:</b><br>
        - Word Cloud hiển thị các từ khóa phổ biến trong tin tức kinh tế, kích thước từ càng lớn thể hiện tần suất xuất hiện càng cao.<br>
        - Biểu đồ thanh Top 10 Keywords giúp so sánh định lượng giữa các từ khóa.<br>
        - Tính năng so sánh nhiều năm cho phép theo dõi xu hướng thay đổi của các chủ đề nóng qua thời gian.<br>
        - Các stopwords tiếng Việt đã được loại bỏ để tập trung vào từ khóa có ý nghĩa.
        </div>
        """,
        unsafe_allow_html=True,
    )
