# ======================================================
# 📰 ui/news_tab.py — Tab tin tức từ nhiều nguồn
# ======================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import feedparser
import json


# ======================================================
# 🔧 HÀM PHỤ TRỢ
# ======================================================
def convert_relative_date(relative_date):
    """Chuyển đổi thời gian tương đối thành thời gian thực"""
    try:
        if "minute" in relative_date:
            minutes = int(relative_date.split()[0])
            return datetime.now() - timedelta(minutes=minutes)
        elif "hour" in relative_date:
            hours = int(relative_date.split()[0])
            return datetime.now() - timedelta(hours=hours)
        elif "day" in relative_date:
            days = int(relative_date.split()[0])
            return datetime.now() - timedelta(days=days)
        else:
            return datetime.now()
    except Exception as e:
        st.warning(f"Error parsing date: {e}")
        return datetime.now()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss_news(source="vnexpress", max_articles=5):
    """Lấy tin từ RSS Feed - Phương pháp đáng tin cậy hơn"""
    
    # Special handling for vnEconomy - use web scraping instead
    if source == "vnEconomy":
        return scrape_vneconomy_news(max_articles)
    
    rss_urls = {
        "vnexpress": "https://vnexpress.net/rss/kinh-doanh.rss",
        "cafef": "https://cafef.vn/thi-truong-chung-khoan.rss",
        "vietstock": "https://vietstock.vn/rss/tintuc.rss"
    }
    
    if source not in rss_urls:
        return []
    
    # Get URL(s) for the source
    urls = rss_urls[source]
    if not isinstance(urls, list):
        urls = [urls]
    
    # Try each URL until one works
    for url_index, url in enumerate(urls):
        try:
            # Enhanced headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # Dùng requests để lấy RSS với timeout ngắn
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Parse RSS
            feed = feedparser.parse(response.content)
            
            # Check if feed has entries
            if not feed.entries:
                if url_index < len(urls) - 1:
                    continue  # Try next URL
                else:
                    st.warning(f"⚠️ Không tìm thấy bài viết từ {source}")
                    return []
            
            news_data = []
            for entry in feed.entries[:max_articles]:
                try:
                    title = entry.title if hasattr(entry, 'title') else "No Title"
                    link = entry.link if hasattr(entry, 'link') else ""
                    
                    # Parse date
                    if hasattr(entry, 'published'):
                        date = entry.published
                    elif hasattr(entry, 'updated'):
                        date = entry.updated
                    else:
                        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Get content
                    content = ""
                    if hasattr(entry, 'summary'):
                        content = BeautifulSoup(entry.summary, 'html.parser').get_text(strip=True)
                    elif hasattr(entry, 'description'):
                        content = BeautifulSoup(entry.description, 'html.parser').get_text(strip=True)
                    else:
                        content = "Nội dung đang được cập nhật..."
                    
                    news_data.append({
                        "title": title,
                        "date": date,
                        "content": content[:500] + "..." if len(content) > 500 else content,
                        "link": link,
                        "source": source.upper()
                    })
                except Exception:
                    continue
            
            if news_data:
                return news_data
            elif url_index < len(urls) - 1:
                continue  # Try next URL
                
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            if url_index < len(urls) - 1:
                continue  # Try next URL
            st.warning(f"⚠️ Không thể tải RSS từ {source}: {error_msg}")
        except requests.exceptions.Timeout:
            if url_index < len(urls) - 1:
                continue  # Try next URL
            st.warning(f"⚠️ Timeout khi tải RSS từ {source}")
        except requests.exceptions.ConnectionError:
            if url_index < len(urls) - 1:
                continue  # Try next URL
            st.warning(f"⚠️ Lỗi kết nối đến {source}")
        except Exception as e:
            if url_index < len(urls) - 1:
                continue  # Try next URL
            st.warning(f"⚠️ Không thể tải RSS từ {source}: {str(e)[:80]}")
    
    return []


@st.cache_data(ttl=300, show_spinner=False)
def scrape_vneconomy_news(max_articles=5):
    """
    Web scraping cho vnEconomy khi RSS không hoạt động
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive'
        }
        
        # Try different sections
        urls_to_try = [
            "https://vneconomy.vn/kinh-te.htm",
            "https://vneconomy.vn/chung-khoan.htm",
            "https://vneconomy.vn"
        ]
        
        for base_url in urls_to_try:
            try:
                response = requests.get(base_url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                news_data = []
                
                # Find article containers - vnEconomy uses different classes
                # Try multiple possible selectors
                article_selectors = [
                    'div.story',
                    'div.story-item',
                    'article.story',
                    'div.news-item',
                    'div.item-news'
                ]
                
                articles = []
                for selector in article_selectors:
                    articles = soup.select(selector)
                    if articles:
                        break
                
                if not articles:
                    # Fallback: find any links that look like articles
                    articles = soup.find_all('a', href=True)
                    articles = [a for a in articles if '/tin-tuc/' in a.get('href', '') or '/kinh-te/' in a.get('href', '')][:max_articles * 2]
                
                for article in articles[:max_articles * 2]:
                    if len(news_data) >= max_articles:
                        break
                    
                    try:
                        # Extract title
                        title_elem = article.find('h3') or article.find('h2') or article.find('a')
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text(strip=True)
                        if not title or len(title) < 10:
                            continue
                        
                        # Extract link
                        link_elem = article.find('a') if article.name != 'a' else article
                        link = link_elem.get('href', '') if link_elem else ''
                        if link and not link.startswith('http'):
                            link = f"https://vneconomy.vn{link}"
                        
                        # Extract date
                        time_elem = article.find('time') or article.find('span', class_=['time', 'date', 'published'])
                        date = time_elem.get_text(strip=True) if time_elem else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Extract description
                        desc_elem = article.find('p') or article.find('div', class_=['description', 'desc', 'summary'])
                        content = desc_elem.get_text(strip=True) if desc_elem else "Đọc thêm tại vneconomy.vn"
                        
                        if len(content) < 20:
                            content = f"{title[:100]}... Đọc thêm tại vneconomy.vn"
                        
                        news_data.append({
                            "title": title,
                            "date": date,
                            "content": content[:500] + "..." if len(content) > 500 else content,
                            "link": link,
                            "source": "VNECONOMY "
                        })
                    except Exception:
                        continue
                
                if news_data:
                    return news_data
                    
            except Exception:
                continue
        
        return []
        
    except Exception as e:
        st.warning(f"⚠️ Không thể scrape vnEconomy: {str(e)[:80]}")
        return []


@st.cache_data(ttl=300, show_spinner=False)  # Cache 5 phút
def scrape_investing_news(page_num, max_articles=5):
    """
    Scrape tin tức từ Investing.com
    
    Args:
        page_num: Số trang cần crawl
        max_articles: Số bài viết tối đa cần lấy
    
    Returns:
        List[dict]: Danh sách tin tức
    """
    # URL đúng cho Investing.com stock market news
    if page_num == 1:
        url = "https://www.investing.com/news/stock-market-news"
    else:
        url = f"https://www.investing.com/news/stock-market-news/{page_num}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Không thể kết nối đến Investing.com: {str(e)[:100]}")
        st.info("💡 Có thể do: (1) Mạng bị chặn, (2) Website đang bảo trì, (3) Cần VPN")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('div', class_='news-analysis-v2_content__z0iLP w-full text-xs sm:flex-1')

    news_data = []
    for article in articles:
        if len(news_data) >= max_articles:
            break
            
        try:
            # Lấy tiêu đề
            title_elem = article.find(
                'a',
                class_='text-inv-blue-500 hover:text-inv-blue-500 hover:underline focus:text-inv-blue-500 focus:underline whitespace-normal text-sm font-bold leading-5 !text-[#181C21] sm:text-base sm:leading-6 lg:text-lg lg:leading-7'
            )
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)

            # Lấy thời gian
            time_elem = article.find('time')
            if time_elem:
                date_text = time_elem.get_text(strip=True)
                if "ago" in date_text:
                    date = convert_relative_date(date_text).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date = date_text
            else:
                date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Lấy liên kết bài viết chi tiết
            link = title_elem.get('href', '')
            if link.startswith("http"):
                full_link = link
            else:
                full_link = f"https://www.investing.com{link}"

            # Lấy nội dung bài viết chi tiết
            content = "Loading..."
            try:
                detail_response = requests.get(full_link, headers=headers, timeout=10)
                detail_response.raise_for_status()
                detail_soup = BeautifulSoup(detail_response.content, 'html.parser')
                content_div = detail_soup.find('div', class_='article_WYSIWYG__O0uhw article_articlePage__UMz3q text-[18px] leading-8')
                content = content_div.get_text(strip=True) if content_div else "No Content Available"
            except requests.exceptions.RequestException as e:
                content = f"Error retrieving content: {e}"

            news_data.append({
                "title": title,
                "date": date,
                "content": content,
                "link": full_link
            })
            
        except Exception as e:
            st.warning(f"⚠️ Error processing article: {e}")
            continue

    return news_data


# ======================================================
# 📰 RENDER TAB NEWS
# ======================================================
def render(ticker: str = None):
    """Hiển thị tab tin tức từ nhiều nguồn"""
    
    st.header("📰 Tin tức Thị trường Chứng khoán")
    
    # Chọn nguồn tin
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <p style='color:#94a3b8'>
        Tin tức mới nhất về thị trường chứng khoán Việt Nam từ nhiều nguồn tin uy tín.
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        news_source = st.selectbox(
            "📡 Chọn nguồn:",
            ["vnexpress", "cafef", "vietstock", "vnEconomy"],
            format_func=lambda x: {
                "vnexpress": "VnExpress",
                "cafef": "CafeF", 
                "vietstock": "VietStock",
                "vnEconomy": "VnEconomy"
            }.get(x, x)
        )
    
    # Khởi tạo session state cho số trang
    if 'news_current_page' not in st.session_state:
        st.session_state.news_current_page = 1
    
    # Tổng số trang
    total_pages = 10
    
    # ======================================================
    # 🎛️ ĐIỀU HƯỚNG TRANG
    # ======================================================
    st.divider()
    
    # Layout điều hướng
    col1, col2, col3 = st.columns([1, 7, 1])
    
    # Nút Previous
    with col1:
        if st.button("⬅️ Previous", use_container_width=True) and st.session_state.news_current_page > 1:
            st.session_state.news_current_page -= 1
            st.rerun()
    
    # Chọn trang
    with col2:
        st.markdown(
            "<div style='text-align: center; font-size: 16px; margin-bottom: 5px;'><b>📄 Go to page:</b></div>",
            unsafe_allow_html=True,
        )
        selected_page = st.number_input(
            "",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.news_current_page,
            step=1,
            label_visibility="collapsed",
            key="news_page_selector"
        )
        if selected_page != st.session_state.news_current_page:
            st.session_state.news_current_page = selected_page
            st.rerun()
    
    # Nút Next
    with col3:
        if st.button("Next ➡️", use_container_width=True) and st.session_state.news_current_page < total_pages:
            st.session_state.news_current_page += 1
            st.rerun()
    
    # Hiển thị số trang hiện tại
    st.info(f"📖 **Trang {st.session_state.news_current_page}** / {total_pages}")
    
    # ======================================================
    # 📊 LẤY VÀ HIỂN THỊ TIN TỨC
    # ======================================================
    page = st.session_state.news_current_page
    
    # Lấy tin tức dựa trên nguồn được chọn
    with st.spinner(f"🔍 Đang tải tin tức từ {news_source.upper()}..."):
        news = fetch_rss_news(news_source, max_articles=5)
    
    if not news:
        st.error(f"❌ Không thể tải tin tức từ nguồn {news_source.upper()}")
        
        # Hiển thị hướng dẫn khắc phục
        st.markdown("""
        ### 🔧 Nguyên nhân có thể:
        
        1. **🌐 Kết nối mạng**: Kiểm tra internet của bạn
        2. **🚫 Website chặn**: Nguồn tin có thể chặn request tự động
        3. **🔒 Firewall/Antivirus**: Có thể đang chặn kết nối
        4. **⏱️ Timeout**: Server phản hồi quá chậm
        
        ### 💡 Giải pháp:
        
        - **Thử nguồn khác**: Chọn nguồn tin khác trong dropdown ở trên
        - Refresh lại trang sau vài giây
        - Kiểm tra kết nối internet
        """)
        
        return  # Dừng execution nếu không có tin tức
    else:
        # Hiển thị từng bài viết
        for index, item in enumerate(news, start=1):
            with st.container():
                # Card-style container
                # Display source badge
                source_badge = item.get('source', 'UNKNOWN')
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                    border-left: 4px solid #22c55e;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                '>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h4 style='color: #166534; margin: 0 0 10px 0; flex: 1;'>📰 Bài {index}: {item['title']}</h4>
                        <span style='background: #22c55e; color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;'>{source_badge}</span>
                    </div>
                    <p style='color: #6b7280; font-size: 14px; margin: 0;'>
                        📅 <b>Đăng lúc:</b> {item['date']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Nút đọc thêm
                with st.expander("📖 Đọc nội dung đầy đủ"):
                    st.markdown(f"**🔗 Link:** [{item['link']}]({item['link']})")
                    st.markdown("---")
                    st.write(item['content'])
                
                st.markdown("<br>", unsafe_allow_html=True)
    
    # ======================================================
    # 💡 HƯỚNG DẪN
    # ======================================================
    st.divider()
    with st.expander("💡 Hướng dẫn sử dụng"):
        st.markdown("""
        ### Cách sử dụng tab News:
        
        1. **Chọn nguồn tin**: Dropdown ở trên cùng để chọn nguồn (VnExpress, CafeF, v.v.)
        2. **Đọc nội dung**: Click vào "Đọc nội dung đầy đủ" để xem chi tiết bài viết
        3. **Cache**: Dữ liệu được cache 5 phút để tăng tốc độ load
        4. **Refresh**: Đợi 5 phút hoặc reload trang (Ctrl+R) để cập nhật tin mới
        
        ### Nguồn tin khả dụng:
        - **VnExpress**: Tin kinh doanh từ VnExpress.net
        - **CafeF**: Tin thị trường chứng khoán từ CafeF.vn
        - **VietStock**: Tin tức từ VietStock.vn
        - **VnEconomy**: Tin kinh tế từ VnEconomy.vn (sử dụng web scraping)
        """)