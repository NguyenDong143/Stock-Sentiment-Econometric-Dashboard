# ======================================================
# 📰 ui/news_tab.py — Tab tin tức từ nhiều nguồn
# ======================================================
"""
News Tab Module - Hiển thị tin tức chứng khoán từ nhiều nguồn

Features:
- RSS feed parsing (VnExpress, CafeF, VietStock)
- Web scraping (vnEconomy, Investing.com)
- AI sentiment analysis using PhoBERT
- Smart caching và retry logic
- Pagination support
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import feedparser
import json
import time
import math
import numbers
import re
import logging
from email.utils import parsedate_to_datetime
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

# Import PhoBERT sentiment analysis
from models.sentiment_phobert import analyze_sentiment

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================
# 🔧 CONSTANTS & CONFIGURATION
# ======================================================

# Keywords để lọc tin tức về chứng khoán Việt Nam
VN_STOCK_KEYWORDS = [
    "chứng khoán", "thị trường việt nam", "thị trường chứng khoán",
    "vn-index", "vnindex", "vn30", "vni", "hose", "hnx", "upcom",
    "vietstock", "doanh nghiệp niêm yết", "cổ phiếu",
    "ssi", "vcb", "vic", "vnm", "tcbs", "vcbs"
]

# Keywords để loại trừ tin tức không liên quan
EXCLUDED_TOPIC_KEYWORDS = [
    "crypto", "bitcoin", "ethereum", "blockchain", "forex", "fed",
    "nasdaq", "dow jones", "s&p", "us market", "wall street",
    "goldman sachs", "chứng khoán mỹ", "trái phiếu mỹ",
    "tiền ảo", "tiền điện tử"
]

# RSS Feed URLs
RSS_FEEDS = {
    "vnexpress": ["https://vnexpress.net/rss/kinh-doanh.rss"],
    "cafef": ["https://cafef.vn/thi-truong-chung-khoan.rss"],
    "vietstock": [
        "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
        "https://vietstock.vn/739/chung-khoan/giao-dich-noi-bo.rss",
        "https://vietstock.vn/741/chung-khoan/niem-yet.rss"
    ]
}

# Patterns
VNECONOMY_ARTICLE_SLUG = re.compile(r"^/[\w\-/]+-e\d+\.htm$")

# Request configuration
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CACHE_TTL = 300  # 5 minutes

# Headers cho web requests
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache'
}

# ======================================================
# 🔧 HELPER FUNCTIONS - Date & Time
# ======================================================

def convert_relative_date(relative_date: str) -> datetime:
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


def is_vietnam_stock_article(title: str, content: str) -> bool:
    """Kiểm tra bài viết có liên quan đến thị trường chứng khoán Việt Nam."""
    combined_text = f"{title or ''} {content or ''}".lower()
    if any(excluded in combined_text for excluded in EXCLUDED_TOPIC_KEYWORDS):
        return False
    return any(keyword in combined_text for keyword in VN_STOCK_KEYWORDS)


def format_display_date(date_value):
    """Định dạng thời gian thành chuỗi thân thiện DD/MM/YYYY - HH:MM"""
    try:
        if isinstance(date_value, datetime):
            dt = date_value
        elif isinstance(date_value, numbers.Number):
            timestamp = float(date_value)
            if timestamp > 1e12:
                timestamp /= 1000  # vnstock trả về millisecond
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(date_value, time.struct_time):
            dt = datetime.fromtimestamp(time.mktime(date_value))
        elif isinstance(date_value, str):
            stripped_value = date_value.strip()
            if stripped_value.isdigit():
                timestamp = float(stripped_value)
                if timestamp > 1e12:
                    timestamp /= 1000
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = parsedate_to_datetime(stripped_value)
        else:
            dt = datetime.now()

        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)

        return dt.strftime("%d/%m/%Y - %H:%M")
    except Exception:
        if isinstance(date_value, str) and date_value:
            return date_value
        return datetime.now().strftime("%d/%m/%Y - %H:%M")


# ======================================================
# 🤖 AI SENTIMENT ANALYSIS
# ======================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ai_sentiment(text: str) -> Tuple[str, float]:
    """
    Phân tích sentiment sử dụng PhoBERT model
    
    Args:
        text: Văn bản cần phân tích
        
    Returns:
        Tuple[str, float]: (sentiment_label, confidence_score)
    """
    try:
        result = analyze_sentiment(text)
        
        if result and isinstance(result, dict):
            # PhoBERT trả về dict với keys như 'NEG', 'NEU', 'POS'
            # Tìm label có score cao nhất
            label = max(result, key=result.get)
            score = result[label]
            
            # Map PhoBERT labels to our labels
            sentiment_map = {
                'POS': 'positive',
                'NEG': 'negative', 
                'NEU': 'neutral',
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral'
            }
            
            return sentiment_map.get(label, 'neutral'), float(score)
        return 'neutral', 0.5
        
    except Exception as e:
        logger.warning(f"AI sentiment analysis failed: {e}. Falling back to keyword-based.")
        return get_keyword_based_sentiment(text)


def get_keyword_based_sentiment(text: str) -> Tuple[str, float]:
    """
    Phân tích sentiment dựa trên keywords (fallback method)
    
    Args:
        text: Văn bản cần phân tích
        
    Returns:
        Tuple[str, float]: (sentiment_label, confidence_score)
    """
    positive_keywords = ["tăng", "hồi phục", "lãi", "tang", "hoi phuc", "lai", "tích cực", "khởi sắc"]
    negative_keywords = ["giảm", "bán tháo", "lỗ", "giam", "ban thao", "lo", "tiêu cực", "sụt giảm"]
    
    text_lower = text.lower()
    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
    
    if pos_count > neg_count:
        return 'positive', min(0.6 + (pos_count * 0.1), 0.9)
    elif neg_count > pos_count:
        return 'negative', min(0.6 + (neg_count * 0.1), 0.9)
    else:
        return 'neutral', 0.5


def get_news_sentiment_styles(title: str, content: str, use_ai: bool = True) -> Dict[str, str]:
    """
    Xác định sentiment và style cho tin tức
    
    Args:
        title: Tiêu đề tin tức
        content: Nội dung tin tức
        use_ai: Có sử dụng AI sentiment analysis không
        
    Returns:
        Dict với border, background, label, sentiment, confidence
    """
    # Phân tích sentiment dựa trên title
    text = title or content or ""
    
    # Sử dụng AI hoặc keyword-based sentiment
    if use_ai:
        sentiment, confidence = get_ai_sentiment(text)  # Phân tích toàn bộ title
    else:
        sentiment, confidence = get_keyword_based_sentiment(text)
    
    styles = {
        "positive": {
            "border": "#22c55e",
            "background": "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
            "label": "Tin tích cực",
            "icon": "📈"
        },
        "negative": {
            "border": "#ef4444",
            "background": "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)",
            "label": "Tin tiêu cực",
            "icon": "📉"
        },
        "neutral": {
            "border": "#d97706",
            "background": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "label": "Tin trung lập",
            "icon": "📊"
        }
    }
    
    result = styles[sentiment].copy()
    result['sentiment'] = sentiment
    result['confidence'] = confidence
    return result


# ======================================================
# 🌐 HTTP REQUEST UTILITIES
# ======================================================

def make_request_with_retry(url: str, headers: Dict = None, max_retries: int = MAX_RETRIES) -> requests.Response:
    """
    Thực hiện HTTP request với retry logic
    
    Args:
        url: URL cần request
        headers: Custom headers
        max_retries: Số lần retry tối đa
        
    Returns:
        Response object
        
    Raises:
        requests.RequestException: Khi tất cả retry đều thất bại
    """
    if headers is None:
        headers = DEFAULT_HEADERS.copy()
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                headers=headers, 
                timeout=REQUEST_TIMEOUT, 
                allow_redirects=True
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries} for {url}")
            
        except requests.exceptions.HTTPError as e:
            # Don't retry on 4xx errors (client errors)
            if 400 <= e.response.status_code < 500:
                raise
            last_exception = e
            logger.warning(f"HTTP error {e.response.status_code} on attempt {attempt + 1}/{max_retries}")
            
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}")
        
        # Exponential backoff
        if attempt < max_retries - 1:
            sleep_time = RETRY_DELAY * (2 ** attempt)
            time.sleep(sleep_time)
    
    # All retries failed
    raise last_exception


# ======================================================
# 📡 RSS FEED PARSER
# ======================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_rss_news(source: str = "vnexpress", max_articles: int = 5) -> List[Dict]:
    """
    Lấy tin tức từ RSS Feed với error handling và retry logic
    
    Args:
        source: Nguồn tin (vnexpress, cafef, vietstock, vnEconomy)
        max_articles: Số lượng bài viết tối đa
        
    Returns:
        List[Dict]: Danh sách tin tức
    """
    # Special handling for vnEconomy - use web scraping instead
    if source == "vnEconomy":
        return scrape_vneconomy_news(max_articles)
    
    if source not in RSS_FEEDS:
        logger.error(f"Unknown news source: {source}")
        return []
    
    urls = RSS_FEEDS[source]
    aggregated_news = []
    errors = []

    # Try each URL and accumulate until we have enough articles
    for url_index, url in enumerate(urls):
        try:
            # Enhanced headers for RSS
            headers = DEFAULT_HEADERS.copy()
            headers['Accept'] = 'application/rss+xml, application/xml, text/xml, */*'
            
            # Fetch RSS with retry logic
            response = make_request_with_retry(url, headers=headers)
            
            # Parse RSS
            feed = feedparser.parse(response.content)
            
            # Check if feed has entries
            if not feed.entries:
                errors.append(f"Không tìm thấy bài viết từ {url}")
                continue
            
            # Process each entry
            for entry in feed.entries:
                if len(aggregated_news) >= max_articles:
                    break
                    
                try:
                    # Extract title
                    title = entry.title if hasattr(entry, 'title') else "No Title"
                    link = entry.link if hasattr(entry, 'link') else ""
                    
                    # Parse date with multiple fallbacks
                    date = _extract_entry_date(entry)
                    
                    # Get content
                    content = _extract_entry_content(entry)
                    
                    # Normalize content length
                    normalized_content = content[:500] + "..." if len(content) > 500 else content
                    
                    # Filter Vietnam stock articles only
                    if not is_vietnam_stock_article(title, normalized_content):
                        continue

                    aggregated_news.append({
                        "title": title,
                        "date": date,
                        "content": normalized_content,
                        "link": link,
                        "source": source.upper()
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing entry: {e}")
                    continue
            
            if len(aggregated_news) >= max_articles:
                break
                
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            errors.append(f"Lỗi HTTP từ {url}: {error_msg}")
            logger.error(f"HTTP error fetching {url}: {error_msg}")
            
        except requests.exceptions.Timeout:
            errors.append(f"Timeout khi tải {url}")
            logger.error(f"Timeout fetching {url}")
            
        except requests.exceptions.ConnectionError as e:
            errors.append(f"Lỗi kết nối đến {url}")
            logger.error(f"Connection error fetching {url}: {e}")
            
        except Exception as e:
            errors.append(f"Lỗi không xác định: {str(e)[:80]}")
            logger.error(f"Unexpected error fetching {url}: {e}")

    # Return results
    if aggregated_news:
        logger.info(f"Successfully fetched {len(aggregated_news)} articles from {source}")
        return aggregated_news[:max_articles]
    
    # Show errors if no articles found
    if errors:
        error_summary = f"⚠️ Không thể tải RSS từ {source}:\n" + "\n".join(f"• {err}" for err in errors[:3])
        st.warning(error_summary)
    else:
        st.warning(f"⚠️ Không thể tải RSS từ {source}")
    
    return []


def _extract_entry_date(entry) -> str:
    """Extract and format date from RSS entry"""
    published_struct = getattr(entry, 'published_parsed', None)
    updated_struct = getattr(entry, 'updated_parsed', None)
    
    if published_struct:
        return format_display_date(published_struct)
    elif updated_struct:
        return format_display_date(updated_struct)
    elif hasattr(entry, 'published'):
        return format_display_date(entry.published)
    elif hasattr(entry, 'updated'):
        return format_display_date(entry.updated)
    else:
        return format_display_date(datetime.now())


def _extract_entry_content(entry) -> str:
    """Extract content from RSS entry"""
    if hasattr(entry, 'summary'):
        return BeautifulSoup(entry.summary, 'html.parser').get_text(strip=True)
    elif hasattr(entry, 'description'):
        return BeautifulSoup(entry.description, 'html.parser').get_text(strip=True)
    else:
        return "Nội dung đang được cập nhật..."


# ======================================================
# 🕷️ WEB SCRAPING
# ======================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def scrape_vneconomy_news(max_articles: int = 5) -> List[Dict]:
    """
    Web scraping cho vnEconomy với improved error handling
    
    Args:
        max_articles: Số lượng bài viết tối đa
        
    Returns:
        List[Dict]: Danh sách tin tức
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive'
        }
        
        base_section = "https://vneconomy.vn/chung-khoan.htm"
        max_section_pages = 5  # crawl deeper pages to get đủ bài liên quan chứng khoán
        urls_to_try = []

        for page in range(1, max_section_pages + 1):
            if page == 1:
                urls_to_try.append(base_section)
            else:
                urls_to_try.append(f"{base_section}?p={page}")

        # Fallback pages bổ sung thêm bối cảnh kinh tế Việt Nam nếu trang chính thiếu bài
        urls_to_try.extend([
            "https://vneconomy.vn/kinh-te.htm",
            "https://vneconomy.vn"
        ])
        
        collected_news = []
        seen_links = set()

        for base_url in urls_to_try:
            try:
                response = requests.get(base_url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                page_news = []
                
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
                
                for article in articles[:max_articles * 3]:
                    if len(collected_news) >= max_articles:
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
                        raw_date = time_elem.get_text(strip=True) if time_elem else datetime.now()
                        date = format_display_date(raw_date) if raw_date else format_display_date(datetime.now())
                        
                        # Extract description
                        desc_elem = article.find('p') or article.find('div', class_=['description', 'desc', 'summary'])
                        content = desc_elem.get_text(strip=True) if desc_elem else "Đọc thêm tại vneconomy.vn"
                        
                        if len(content) < 20:
                            content = f"{title[:100]}... Đọc thêm tại vneconomy.vn"
                        
                        normalized_content = content[:500] + "..." if len(content) > 500 else content

                        passes_filter = is_vietnam_stock_article(title, normalized_content)
                        lower_text = f"{title} {normalized_content}".lower()
                        if not passes_filter:
                            if (link.startswith("https://vneconomy.vn/chung-khoan") or link.startswith("/chung-khoan") or "chung-khoan" in base_url.lower()) and not any(excluded in lower_text for excluded in EXCLUDED_TOPIC_KEYWORDS):
                                passes_filter = True
                        if not passes_filter:
                            continue

                        unique_key = link or title
                        if unique_key in seen_links:
                            continue
                        seen_links.add(unique_key)

                        page_news.append({
                            "title": title,
                            "date": date,
                            "content": normalized_content,
                            "link": link,
                            "source": "VNECONOMY "
                        })
                    except Exception:
                        continue
                
                if len(collected_news) + len(page_news) < max_articles:
                    for anchor in soup.find_all('a', href=True):
                        if len(collected_news) + len(page_news) >= max_articles:
                            break
                        raw_href = anchor.get('href', '')
                        if not raw_href or raw_href.startswith('javascript') or raw_href.startswith('#'):
                            continue
                        if not VNECONOMY_ARTICLE_SLUG.match(raw_href):
                            continue
                        anchor_title = anchor.get_text(strip=True)
                        if not anchor_title or len(anchor_title) < 10:
                            continue
                        link = raw_href if raw_href.startswith('http') else f"https://vneconomy.vn{raw_href}"
                        if link in seen_links:
                            continue

                        placeholder_content = f"Tin nhanh VnEconomy: {anchor_title}. Đọc nội dung chi tiết trên trang gốc."
                        passes_filter = is_vietnam_stock_article(anchor_title, placeholder_content)
                        if not passes_filter:
                            lower_text = anchor_title.lower()
                            if (link.startswith("https://vneconomy.vn/chung-khoan") or raw_href.startswith("/chung-khoan") or "chung-khoan" in base_url.lower()) and not any(excluded in lower_text for excluded in EXCLUDED_TOPIC_KEYWORDS):
                                passes_filter = True
                        if not passes_filter:
                            continue

                        seen_links.add(link)
                        page_news.append({
                            "title": anchor_title,
                            "date": format_display_date(datetime.now()),
                            "content": placeholder_content,
                            "link": link,
                            "source": "VNECONOMY "
                        })

                if page_news:
                    collected_news.extend(page_news)
                    if len(collected_news) >= max_articles:
                        return collected_news[:max_articles]
                    
            except Exception:
                continue
        
        return collected_news
        
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
                    date = format_display_date(convert_relative_date(date_text))
                else:
                    date = format_display_date(date_text)
            else:
                date = format_display_date(datetime.now())

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

            if not is_vietnam_stock_article(title, content):
                continue

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


def render_pagination_controls(total_pages):
    """Hiển thị điều hướng trang ở cuối tab"""
    st.divider()
    spacer_left, control_col, spacer_right = st.columns([1, 2, 1])

    with control_col:
        prev_col, info_col, next_col = st.columns([1, 1, 1], gap="small")

        prev_disabled = st.session_state.news_current_page <= 1
        next_disabled = st.session_state.news_current_page >= total_pages

        if prev_col.button("⬅️", use_container_width=True, disabled=prev_disabled, key="news_prev_btn"):
            st.session_state.news_current_page -= 1
            st.rerun()

        info_col.markdown(
            f"<div style='text-align:center; font-size:16px; font-weight:600;'>Trang {st.session_state.news_current_page} / {total_pages}</div>",
            unsafe_allow_html=True
        )

        if next_col.button("➡️", use_container_width=True, disabled=next_disabled, key="news_next_btn"):
            st.session_state.news_current_page += 1
            st.rerun()


# ======================================================
# 📰 RENDER TAB NEWS
# ======================================================

def render(ticker: str = None):
    """
    Hiển thị tab tin tức từ nhiều nguồn với AI sentiment analysis
    
    Args:
        ticker: Mã cổ phiếu (optional, for future filtering)
    """
    st.header("📰 Tin tức Thị trường Chứng khoán Việt Nam")
    
    # Chọn nguồn tin và settings
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        <p style='color:#94a3b8'>
        Tin tức mới nhất về thị trường chứng khoán Việt Nam với phân tích sentiment AI.
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
            }.get(x, x),
            key="news_source_select"
        )
    
    with col3:
        use_ai_sentiment = st.checkbox(
            "🤖 AI Sentiment",
            value=True,
            help="Sử dụng PhoBERT để phân tích cảm xúc tin tức"
        )
    
    # Khởi tạo session state
    if 'news_current_page' not in st.session_state:
        st.session_state.news_current_page = 1
    
    if 'last_news_source' not in st.session_state:
        st.session_state.last_news_source = news_source
    
    # Reset page khi đổi nguồn
    if st.session_state.last_news_source != news_source:
        st.session_state.news_current_page = 1
        st.session_state.last_news_source = news_source
    
    per_page = 5
    
    # Refresh button
    col_refresh, col_spacer = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Làm mới", key="refresh_news", help="Tải lại tin tức mới"):
            st.cache_data.clear()
            st.rerun()
    
    st.divider()
    
    # ======================================================
    # 📊 LẤY VÀ HIỂN THỊ TIN TỨC
    # ======================================================
    # Progress tracking
    progress_placeholder = st.empty()
    
    with progress_placeholder:
        with st.spinner(f"🔍 Đang tải tin tức từ {news_source.upper()}..."):
            news = fetch_rss_news(news_source, max_articles=50)
    
    progress_placeholder.empty()
    
    if not news:
        st.error(f"❌ Không thể tải tin tức từ nguồn {news_source.upper()}")
        
        # Hiển thị hướng dẫn khắc phục
        with st.expander("🔧 Hướng dẫn khắc phục", expanded=True):
            st.markdown("""
            ### Nguyên nhân có thể:
            
            - 🌐 **Kết nối mạng**: Kiểm tra internet của bạn
            - 🚫 **Website chặn**: Nguồn tin có thể chặn request tự động
            - 🔒 **Firewall/Antivirus**: Có thể đang chặn kết nối
            - ⏱️ **Timeout**: Server phản hồi quá chậm
            
            ### Giải pháp:
            
            1. **Thử nguồn khác**: Chọn nguồn tin khác trong dropdown ở trên
            2. **Làm mới**: Click nút "🔄 Làm mới" ở trên
            3. **Kiểm tra kết nối**: Đảm bảo internet hoạt động bình thường
            """)
        
        return  # Dừng execution nếu không có tin tức
    
    # Phân trang
    total_pages = max(1, math.ceil(len(news) / per_page))
    current_page = min(st.session_state.news_current_page, total_pages)
    
    if current_page != st.session_state.news_current_page:
        st.session_state.news_current_page = current_page
        st.rerun()
    
    start_idx = (current_page - 1) * per_page
    page_news = news[start_idx:start_idx + per_page]
    
    if not page_news and current_page > 1:
        st.session_state.news_current_page = 1
        st.rerun()
    
    # Hiển thị từng bài viết với sentiment analysis
    for index, item in enumerate(page_news, start=start_idx + 1):
        sentiment_styles = get_news_sentiment_styles(
            item['title'], 
            item['content'],
            use_ai=use_ai_sentiment
        )
        
        border_color = sentiment_styles['border']
        background_style = sentiment_styles['background']
        sentiment_label = sentiment_styles['label']
        sentiment_icon = sentiment_styles.get('icon', '📊')
        confidence = sentiment_styles.get('confidence', 0.0)
        
        # Tạo title link
        title_link = f"<a href='{item['link']}' target='_blank' style='color:#0f172a; text-decoration:none; hover:text-decoration:underline;'>{item['title']}</a>"
        
        # Hiển thị sentiment badge với confidence
        confidence_pct = int(confidence * 100)
        sentiment_badge = f"{sentiment_icon} {sentiment_label}"
        if use_ai_sentiment:
            sentiment_badge += f" ({confidence_pct}%)"

        with st.container():
            st.markdown(f"""
            <div style='
                background: {background_style};
                border-left: 5px solid {border_color};
                padding: 18px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            '>
                <div style='display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 12px;'>
                    <h4 style='color: #0f172a; margin: 0; flex: 1; line-height: 1.4;'>
                        {title_link}
                    </h4>
                    <span style='
                        font-size: 11px; 
                        font-weight: 600; 
                        color: {border_color}; 
                        padding: 6px 12px; 
                        border: 1.5px solid {border_color}; 
                        border-radius: 20px;
                        white-space: nowrap;
                        background: rgba(255,255,255,0.7);
                    '>
                        {sentiment_badge}
                    </span>
                </div>
                <p style='color: #6b7280; font-size: 13px; margin: 8px 0 0 0;'>
                    📅 <b>Đăng lúc:</b> {item['date']} | 📰 <b>Nguồn:</b> {item.get('source', news_source.upper())}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Nội dung
            st.markdown(f"<p style='color:#ffffff; line-height:1.6;'>{item['content']}</p>", unsafe_allow_html=True)
            
            # Link đọc thêm
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Pagination controls
    render_pagination_controls(total_pages)