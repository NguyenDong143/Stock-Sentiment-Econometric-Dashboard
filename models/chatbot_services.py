"""
Chatbot Services - Portfolio AI Assistant using Google Gemini API
WITH STREAMING SUPPORT + VNDirect Timeout Handling + Symbol Filtering + Multi-API Key Rotation
OPTIMIZED: Concurrent data fetching, Enhanced prompts, Few-shot learning
"""

import google.generativeai as genai
from google.generativeai import types
from typing import Optional, List, Dict, Iterator, Tuple
import re
import sys
import os
import logging
import streamlit as st
import concurrent.futures
from functools import lru_cache

# Import VNDIRECT API và History Manager
from utils.vndirect_api import get_vndirect_api
from utils.chat_history_manager import ChatHistoryManager
from utils.data_loader import load_price_data, load_sentiment_data, load_realtime_price_quote

logger = logging.getLogger(__name__)

# ====================================================================
# CONSTANTS FOR OPTIMIZATION
# ====================================================================
CACHE_TTL_REALTIME = 30      # Cache giá realtime 30s
CACHE_TTL_TECHNICAL = 600    # Cache phân tích kỹ thuật 10 phút
CACHE_TTL_SENTIMENT = 600    # Cache sentiment 10 phút
MAX_CONCURRENT_WORKERS = 4   # Số luồng song song tối đa
DATA_FETCH_TIMEOUT = 5       # Timeout cho mỗi data fetch (giây)


@st.cache_resource(show_spinner=False)
def _initialize_genai_model(api_key: str) -> str:
    """Cache việc khởi tạo Gemini model - ULTRA FAST (No Network Check)"""
    genai.configure(api_key=api_key)
    
    # Trả về model name ngay lập tức, không check network
    # Gemini 1.5 Flash là lựa chọn tốt nhất hiện tại cho tốc độ/giá/performance
    return "gemini-1.5-flash"


class PortfolioChatbot:
    """AI Chatbot for portfolio and stock analysis using Google Gemini"""

    def __init__(self, api_key: str, session_id: str = "default", auto_load: bool = True):
        # Use cached model initialization
        self.model = _initialize_genai_model(api_key)
        genai.configure(api_key=api_key)

        # Session management
        self.session_id = session_id
        self.history_manager = ChatHistoryManager()
        
        # Khởi tạo chat history
        self.chat_history = []
        
        # Tự động tải lịch sử từ file nếu có
        if auto_load:
            self._load_history_from_file()

        # SYSTEM PROMPT for Vietnamese stock market - ENHANCED với Few-Shot Examples
        self.system_prompt = """Bạn là chuyên gia phân tích thị trường chứng khoán Việt Nam với khả năng:

1. **TỔNG QUAN THỊ TRƯỜNG**: Phân tích VNINDEX, HNXINDEX, thanh khoản, xu hướng
2. **Phân tích GIÁ REALTIME**: Giải thích biến động giá, volume, mức hỗ trợ/kháng cự
3. **Chỉ số kỹ thuật**: RSI, MACD, Bollinger Bands, SMA/EMA, ADX, Stochastic
4. **Phân tích cảm xúc**: Tác động tin tức lên giá cổ phiếu (PhoBERT sentiment)
5. **So sánh ngành**: So sánh hiệu suất giữa các mã cùng ngành
6. **Tư vấn chiến lược**: Phân tích rủi ro/cơ hội dựa trên dữ liệu

**===== VÍ DỤ CÂU TRẢ LỜI CHUẨN =====**

**Q: Thị trường hôm nay thế nào?**
A: 🔺 VNINDEX tăng 12.5 điểm (+1.02%) lên 1,235.50 điểm. Thanh khoản đạt 18,500 tỷ đồng. Blue chips dẫn dắt: VCB +2.1%, BID +1.8%, HPG +1.5%. Xu hướng tích cực ngắn hạn với lực cầu mạnh.

**Q: RSI VCB bao nhiêu?**
A: RSI(14) của VCB đang ở mức 65.3 - vùng trung tính. Chưa quá mua (<70), có thể tiếp tục tăng nếu vượt 70. Hỗ trợ: 92,000 | Kháng cự: 98,000.

**Q: So sánh VCB và BID?**
A: 📊 30 ngày qua: VCB +5.2% vs BID +3.8%. VCB vượt trội với RSI (65) > BID (58). VCB đang mạnh hơn về momentum, phù hợp xu hướng tăng ngắn hạn.

**===== QUY TẮC TRẢ LỜI =====**
- Trả lời ngắn gọn (2-4 câu), DỄ HIỂU, có số liệu cụ thể
- LUÔN dựa vào dữ liệu context được cung cấp (nếu có)
- Nếu có data về blue chips, phân tích xu hướng chung từ đó
- KHÔNG nói "Tôi là AI không thể cung cấp thông tin realtime" - Hãy phân tích dữ liệu được cung cấp
- KHÔNG đưa ra khuyến nghị mua/bán cụ thể
- Phân tích khách quan dựa trên số liệu, không khẳng định tương lai
- Sử dụng emoji phù hợp để tăng tính trực quan (🔺🔻📊📈📉)"""

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

    # ====================================================================
    # MODEL FINDER
    # ====================================================================
    def _find_available_model(self):
        print("🔍 Đang tìm model Gemini khả dụng...")

        # Danh sách models khả dụng (updated theo API v1beta)
        model_list = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-pro-latest",
            "gemini-2.5-pro",
        ]

        last_error = None
        for name in model_list:
            try:
                print(f"Thử: {name}")
                # Sử dụng GenerativeModel để test
                test_model = genai.GenerativeModel(name)
                test_response = test_model.generate_content(
                    "hello",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=10
                    )
                )
                
                # Kiểm tra response có text không
                if hasattr(test_response, 'text'):
                    print(f"✅ Sử dụng: {name}")
                    return name
                    
            except Exception as e:
                # Chỉ print lỗi nếu debug mode
                # print(f"❌ Model {name} không khả dụng: {e}")
                last_error = e
                continue

        # Nếu không model nào hoạt động, raise chi tiết lỗi cuối cùng
        error_msg = f"❌ Không tìm thấy model Gemini khả dụng!"
        if last_error:
            error_msg += f"\nLỗi cuối cùng: {str(last_error)}"
        raise Exception(error_msg)
    
    # Hàm khởi tạo model instance cho mỗi lần gọi (để đảm bảo config)
    def _get_model_instance(self):
        return genai.GenerativeModel(
            self.model,
            system_instruction=self.system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.85,
                top_k=30,
                max_output_tokens=800,  # Tăng lên 800 để response không bị cắt
            )
        )
    # ====================================================================
    # STREAMING RESPONSE - OPTIMIZED với Concurrent Data Fetching
    # ====================================================================
    def generate_response_stream(self, user_message: str, context: Optional[str] = None) -> Iterator[str]:
        """Generate streaming response với concurrent data fetching để tăng tốc"""
        try:
            detected_symbols = self._extract_stock_symbols(user_message)
            
            # Thu thập context đa chiều cho chatbot
            context_blocks = []
            
            # KIỂM TRA CÂU HỎI TỔNG QUAN THỊ TRƯỜNG
            market_data_fetched = False
            if self._is_market_overview_query(user_message) and not detected_symbols:
                # Lấy tổng quan thị trường với retry (giữ nguyên logic cũ cho market overview)
                for retry in range(2):
                    try:
                        api = get_vndirect_api()
                        overview = api.get_market_overview()
                        if overview:
                            vn_data = overview.get('vnindex')
                            hnx_data = overview.get('hnxindex')
                            
                            market_info = "📊 TỔNG QUAN THỊ TRƯỜNG HÔM NAY:\n"
                            if vn_data:
                                change_icon = "🔺" if vn_data['change'] > 0 else "🔻" if vn_data['change'] < 0 else "➡️"
                                market_info += f"{change_icon} **VNINDEX**: {vn_data['price']:,.2f} ({vn_data['change']:+,.2f} | {vn_data['change_percent']:+.2f}%)\n"
                            if hnx_data:
                                change_icon = "🔺" if hnx_data['change'] > 0 else "🔻" if hnx_data['change'] < 0 else "➡️"
                                market_info += f"{change_icon} **HNXINDEX**: {hnx_data['price']:,.2f} ({hnx_data['change']:+,.2f} | {hnx_data['change_percent']:+.2f}%)\n"
                            market_info += f"\n⏰ Cập nhật: {overview.get('time', '')}"
                            
                            context_blocks.append(market_info)
                            market_data_fetched = True
                            
                            # Thêm top movers nếu có
                            top_stocks = ["VCB", "BID", "HPG", "VHM", "FPT"]
                            stock_data = api.get_multiple_stocks(top_stocks)
                            if stock_data:
                                movers = "🏆 TOP BLUE CHIPS:\n"
                                for symbol in top_stocks:
                                    if symbol in stock_data:
                                        data = stock_data[symbol]
                                        icon = "🟢" if data['change'] > 0 else "🔴" if data['change'] < 0 else "🟡"
                                        movers += f"{icon} {symbol}: {data['price']:,.0f} ({data['change_percent']:+.2f}%)\n"
                                context_blocks.append(movers)
                            break
                    except Exception as e:
                        logger.warning(f"Lần thử {retry+1} lấy market overview thất bại: {e}")
                        if retry == 0:
                            import time
                            time.sleep(0.5)  # Giảm từ 1s xuống 0.5s
                
                # FALLBACK: Nếu không lấy được data
                if not market_data_fetched:
                    fallback_msg = """⚠️ **Tạm thời không kết nối được nguồn dữ liệu realtime**

Bạn có thể:
- Kiểm tra trên [TCBS](https://tcinvest.tcbs.com.vn/) hoặc [VNDirect](https://www.vndirect.com.vn/)
- Hoặc hỏi tôi về phân tích kỹ thuật, tin tức, hoặc so sánh cổ phiếu cụ thể"""
                    self.chat_history.append({"user": user_message, "assistant": fallback_msg})
                    self._save_history_to_file(metadata={"type": "fallback_api_error"})
                    yield fallback_msg
                    return
            
            # ===== CONCURRENT DATA FETCHING cho mã cụ thể =====
            if detected_symbols:
                primary_symbol = detected_symbols[0]
                
                # Sử dụng ThreadPoolExecutor để fetch song song
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
                    # Submit tất cả tasks song song
                    futures = {
                        'realtime': executor.submit(self._get_realtime_prices, tuple(detected_symbols)),
                        'technical': executor.submit(self._get_technical_analysis, primary_symbol),
                        'sentiment': executor.submit(self._get_sentiment_summary, primary_symbol),
                        'signals': executor.submit(self._get_trading_signals, primary_symbol),
                    }
                    
                    # Thu thập kết quả với timeout
                    results = {}
                    for key, future in futures.items():
                        try:
                            results[key] = future.result(timeout=DATA_FETCH_TIMEOUT)
                        except (concurrent.futures.TimeoutError, Exception) as e:
                            logger.warning(f"Concurrent fetch {key} thất bại: {e}")
                            results[key] = ""
                
                # Thêm kết quả vào context
                if results.get('realtime'):
                    context_blocks.append(f"📈 GIÁ REALTIME:\n{results['realtime']}")
                if results.get('technical'):
                    context_blocks.append(results['technical'])
                if results.get('sentiment'):
                    context_blocks.append(results['sentiment'])
                if results.get('signals'):
                    context_blocks.append(results['signals'])
            
            realtime_prices_markdown = "\n\n".join(context_blocks) if context_blocks else None
            
            # --- START: LOGIC TÁCH GIÁ TRỰC TIẾP ---
            is_pure_price_query = self._is_pure_price_query(user_message, detected_symbols)
            
            # Nếu là câu hỏi thuần túy về giá, trả về giá trực tiếp và dừng lại
            if is_pure_price_query and realtime_prices_markdown:
                full_response = realtime_prices_markdown
                # Lưu lịch sử và yield toàn bộ
                self.chat_history.append({"user": user_message, "assistant": full_response})
                self._save_history_to_file(metadata={"type": "direct_price"})
                yield full_response
                return
            # --- END: LOGIC TÁCH GIÁ TRỰC TIẾP ---

            # Nếu không phải giá thuần túy, CHUYỂN QUA GEMINI
            model_instance = self._get_model_instance()
            
            # Build context cho Gemini
            context_blocks = []
            if realtime_prices_markdown:
                context_blocks.append(f"📈 GIÁ REALTIME (DÙNG ĐỂ PHÂN TÍCH):\n{realtime_prices_markdown}")
            if context:
                context_blocks.append(context)

            full_context = "\n\n".join(context_blocks) if context_blocks else ""
            
            # Gộp lịch sử chat vào prompt (cách đơn giản nhất cho streaming không dùng ChatSession)
            history_context = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.chat_history])

            full_prompt = f"{history_context}\n\n{full_context}\n\nQ: {user_message}\nA:"

            response = model_instance.generate_content(
                full_prompt,
                stream=True,
                safety_settings=self.safety_settings,
            )

            full_text = ""

            for chunk in response:
                # Kiểm tra xem chunk có candidates không
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    # Kiểm tra finish_reason - SAFETY means blocked
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = str(candidate.finish_reason)
                        if 'SAFETY' in finish_reason.upper():
                            yield "⚠️ Câu trả lời bị chặn bởi bộ lọc an toàn. Vui lòng thử câu hỏi khác."
                            return
                    
                    # Lấy text từ parts
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                full_text += part.text
                                yield part.text
                elif hasattr(chunk, "text") and chunk.text:
                    full_text += chunk.text
                    yield chunk.text
                elif hasattr(chunk, "prompt_feedback"):
                    # Kiểm tra prompt bị block
                    if hasattr(chunk.prompt_feedback, "block_reason"):
                        block_reason = str(chunk.prompt_feedback.block_reason)
                        if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
                            yield f"⚠️ Prompt bị chặn: {block_reason}. Vui lòng thử câu hỏi khác."
                            return
            
            # Nếu không có text nào được tạo
            if not full_text:
                yield "⚠️ Không nhận được phản hồi từ AI. Vui lòng thử lại."
                return

            if full_text:
                self.chat_history.append({
                    "user": user_message,
                    "assistant": full_text
                })
                # Auto-save sau mỗi tin nhắn
                self._save_history_to_file(metadata={"type": "streaming"})

        except Exception as e:
            error_str = str(e)
            logger.error(f"Lỗi streaming: {e}", exc_info=True)
            
            # Xử lý lỗi response bị block (Invalid operation)
            if "Invalid operation" in error_str and "response.text" in error_str:
                yield "⚠️ Phản hồi bị chặn bởi bộ lọc an toàn. Vui lòng thử câu hỏi khác hoặc diễn đạt rõ ràng hơn."
            # Xử lý lỗi API key cụ thể
            elif "403" in error_str and "leaked" in error_str.lower():
                yield "❌ **API Key đã bị vô hiệu hóa!**\n\nAPI key của bạn đã bị Google phát hiện leaked. Vui lòng:\n1. Tạo key mới tại: https://makersuite.google.com/app/apikey\n2. Cập nhật file `config/settings.py`\n3. Restart app"
            elif "403" in error_str or "quota" in error_str.lower() or "429" in error_str:
                # Thử xoay vòng API key
                try:
                    key_manager = get_api_key_manager()
                    if key_manager and key_manager.available_keys_count > 1:
                        if key_manager.mark_current_failed():
                            # Thông báo đang chuyển key
                            status = key_manager.get_status()
                            yield f"🔄 **Đang chuyển sang API key #{status['current_key_index']}/{status['total_keys']}...**\n\nVui lòng thử lại câu hỏi!"
                            # Cập nhật API key cho genai
                            genai.configure(api_key=key_manager.current_key)
                            return
                        else:
                            yield "❌ **Tất cả API keys đã hết quota!**\n\nVui lòng đợi 1 phút hoặc thêm API key mới vào `config/settings.py`"
                            return
                    else:
                        yield "⚠️ **Đã hết quota API**\n\nVui lòng đợi 1 phút hoặc thêm thêm API key vào `GEMINI_API_KEYS` trong `config/settings.py`"
                except Exception as key_error:
                    logger.error(f"Lỗi xoay vòng key: {key_error}")
                    yield "⚠️ **Đã hết quota API**\n\nVui lòng đợi 1 phút hoặc sử dụng API key khác."
            elif "finish_reason" in error_str.lower() or "safety" in error_str.lower():
                yield "⚠️ Phản hồi bị chặn bởi bộ lọc an toàn. Vui lòng thử câu hỏi khác."
            else:
                yield f"⚠️ Lỗi: {error_str[:150]}"

    # ====================================================================
    # NORMAL RESPONSE (Dùng cho test hoặc không streaming)
    # ====================================================================
    def generate_response(self, user_message: str, context: Optional[str] = None) -> str:
        # Tái sử dụng logic của hàm streaming nhưng không stream
        try:
            detected_symbols = self._extract_stock_symbols(user_message)
            
            realtime_prices_markdown = None
            if detected_symbols:
                realtime_prices_markdown = self._get_realtime_prices(tuple(detected_symbols))

            is_pure_price_query = self._is_pure_price_query(user_message, detected_symbols)
            
            # Nếu là câu hỏi thuần túy về giá, trả về giá trực tiếp
            if is_pure_price_query and realtime_prices_markdown:
                full_response = realtime_prices_markdown
                self.chat_history.append({"user": user_message, "assistant": full_response})
                self._save_history_to_file(metadata={"type": "direct_price"})
                return full_response

            # Nếu không phải giá thuần túy, CHUYỂN QUA GEMINI
            model_instance = self._get_model_instance()
            
            context_blocks = []
            if realtime_prices_markdown:
                context_blocks.append(f"📈 GIÁ REALTIME (DÙNG ĐỂ PHÂN TÍCH):\n{realtime_prices_markdown}")
            if context:
                context_blocks.append(context)

            full_context = "\n\n".join(context_blocks) if context_blocks else ""
            
            history_context = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.chat_history])
            full_prompt = f"{history_context}\n\n{full_context}\n\nQ: {user_message}\nA:"


            resp = model_instance.generate_content(
                full_prompt,
                safety_settings=self.safety_settings
            )

            text = resp.text if hasattr(resp, "text") else str(resp)

            self.chat_history.append({"user": user_message, "assistant": text})
            self._save_history_to_file(metadata={"type": "normal"})
            return text

        except Exception as e:
            logger.error(f"Lỗi generate_response: {e}", exc_info=True)
            return f"⚠️ Lỗi: {str(e)[:200]}"

    # ====================================================================
    # SYMBOL EXTRACTOR & QUERY CLASSIFIER
    # ====================================================================
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """
        Detect VN stock codes (3-4 uppercase letters), excluding false positives.
        Enhanced to better detect VNINDEX and market indices.
        """

        pattern = r"\b([A-Z]{3,4})\b"
        matches = re.findall(pattern, text.upper())

        # Words to ignore (Common English words & Vietnamese false positives)
        exclude = {
            "THE", "AND", "FOR", "NOT", "BUT", "CAN", "YOU", "ALL", "NEW", "ARE",
            "TIN", "SAO", "CHO", "KHI", "USD", "ETF", "NAV", "AI", "VN", "GDP",
            "MUA", "BAN", "VAY", "NHA", "CAM", "HOI", "GIA", "NAM", "CHIA",
            "NAY", "QUA", "NGO" 
        }

        unique_symbols = list(set([m for m in matches if m not in exclude]))
        
        # TĂNG CƯỜNG: Phát hiện chỉ số thị trường từ ngữ cảnh
        text_lower = text.lower()
        
        # Aliases cho VNINDEX
        vnindex_keywords = ["vnindex", "vn-index", "vn index", "chỉ số vn", 
                          "thị trường", "hôm nay", "phiên", "tổng quan"]
        
        # Nếu không có mã cụ thể VÀ hỏi về thị trường chung → thêm VNINDEX
        if not unique_symbols and any(kw in text_lower for kw in vnindex_keywords):
            unique_symbols.append("VNINDEX")
        
        # Phát hiện rõ ràng VNINDEX/HNXINDEX
        if any(idx in text_lower for idx in ["vnindex", "hnxindex", "upcom"]):
            if "vnindex" in text_lower and "VNINDEX" not in unique_symbols:
                unique_symbols.append("VNINDEX")
            if "hnxindex" in text_lower and "HNXINDEX" not in unique_symbols:
                unique_symbols.append("HNXINDEX")
        
        return unique_symbols


    def _is_market_overview_query(self, text: str) -> bool:
        """Kiểm tra câu hỏi có phải hỏi tổng quan thị trường không"""
        text_lower = text.lower()
        market_keywords = [
            "thị trường", "vnindex", "vn-index", "vn index", "hnx", "upcom", "chỉ số", 
            "giao dịch", "thanh khoản", "tổng quan", "tình hình",
            "hôm nay", "phiên", "khối ngoại", "thế nào", "ra sao",
            "bây giờ", "hiện tại", "đang"
        ]
        # Kiểm tra từ khóa hoặc câu hỏi ngắn không có mã cổ phiếu cụ thể
        has_keyword = any(keyword in text_lower for keyword in market_keywords)
        is_short_query = len(text.split()) <= 6  # Câu hỏi ngắn thường là hỏi tổng quan
        
        return has_keyword or (is_short_query and not re.findall(r"\b[A-Z]{3}\b", text.upper()))
    
    def _is_pure_price_query(self, text: str, detected_symbols: List[str]) -> bool:
        """Kiểm tra xem câu hỏi có thuần túy là hỏi giá (không cần phân tích) không."""
        if not detected_symbols:
            return False

        # Các từ khóa chỉ hỏi giá
        price_keywords = ["giá", "hiện tại", "bao nhiêu", "là mấy", "bây giờ"]

        # Nếu tìm thấy mã cổ phiếu VÀ từ khóa hỏi giá, VÀ câu hỏi ngắn
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in price_keywords) and len(text.split()) < 10:
            return True
            
        # Ví dụ: "VCB giá", "HPG"
        if len(detected_symbols) >= 1 and len(text.split()) <= 4:
            return True

        return False

    # ====================================================================
    # MARKET DATA ANALYSIS TOOLS FOR CHATBOT
    # ====================================================================
    @st.cache_data(ttl=CACHE_TTL_TECHNICAL, show_spinner=False)
    def _get_technical_analysis(_self, symbol: str) -> str:
        """Lấy phân tích kỹ thuật cơ bản cho chatbot"""
        try:
            df = load_price_data(symbol)
            if df.empty or len(df) < 20:
                return ""
            
            # Tính các chỉ số cơ bản
            current_price = df['close'].iloc[-1]
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else None
            
            # Volume trend
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1
            
            analysis = f"""📊 PHÂN TÍCH KỸ THUẬT {symbol}:
- Giá hiện tại: {current_price:,.0f} VNĐ
- SMA(20): {sma_20:,.0f} | Xu hướng: {'Tăng' if current_price > sma_20 else 'Giảm'}"""
            
            if sma_50:
                analysis += f"\n- SMA(50): {sma_50:,.0f} | Golden Cross: {'Có' if sma_20 > sma_50 else 'Không'}"
            
            if current_rsi:
                rsi_status = "Quá mua" if current_rsi > 70 else "Quá bán" if current_rsi < 30 else "Trung tính"
                analysis += f"\n- RSI(14): {current_rsi:.1f} - {rsi_status}"
            
            analysis += f"\n- Volume: {volume_ratio:.1f}x trung bình ({'Mạnh' if volume_ratio > 1.5 else 'Bình thường'})"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Lỗi phân tích kỹ thuật {symbol}: {e}")
            return ""
    
    @st.cache_data(ttl=CACHE_TTL_SENTIMENT, show_spinner=False)
    def _get_sentiment_summary(_self, symbol: str) -> str:
        """Lấy tóm tắt cảm xúc tin tức"""
        try:
            # Lấy sentiment data từ session state nếu có
            if 'ticker' in st.session_state and st.session_state.ticker == symbol:
                df = load_sentiment_data(
                    ticker=symbol,
                    data_type=st.session_state.get('data_type', 'Content'),
                    time_period=st.session_state.get('time_period', 'After Scandal')
                )
                
                if not df.empty and 'sentiment_label' in df.columns:
                    # Đếm sentiment
                    sentiment_counts = df['sentiment_label'].value_counts()
                    total = len(df)
                    
                    positive = sentiment_counts.get(1, 0)
                    negative = sentiment_counts.get(-1, 0)
                    neutral = sentiment_counts.get(0, 0)
                    
                    return f"""📰 PHÂN TÍCH CẢM XÚC TIN TỨC {symbol}:
- Tổng: {total} bài viết
- Tích cực: {positive} ({positive/total*100:.1f}%)
- Tiêu cực: {negative} ({negative/total*100:.1f}%)
- Trung tính: {neutral} ({neutral/total*100:.1f}%)
- Xu hướng: {'Tích cực' if positive > negative else 'Tiêu cực' if negative > positive else 'Trung tính'}"""
            
            return ""
        except Exception as e:
            logger.error(f"Lỗi phân tích sentiment {symbol}: {e}")
            return ""

    # ====================================================================
    # TRADING SIGNALS - Tín hiệu giao dịch
    # ====================================================================
    @st.cache_data(ttl=CACHE_TTL_TECHNICAL, show_spinner=False)
    def _get_trading_signals(_self, symbol: str) -> str:
        """Lấy tín hiệu giao dịch kỹ thuật"""
        try:
            df = load_price_data(symbol)
            if df.empty or len(df) < 50:
                return ""
            
            signals = []
            
            # RSI Signal
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 30:
                signals.append("🟢 RSI < 30: Tín hiệu MUA (quá bán)")
            elif current_rsi > 70:
                signals.append("🔴 RSI > 70: Tín hiệu BÁN (quá mua)")
            
            # SMA Signal
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                signals.append("🟢 Golden Cross: Xu hướng TĂNG")
            elif current_price < sma_20 < sma_50:
                signals.append("🔴 Death Cross: Xu hướng GIẢM")
            
            # Volume Signal
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            if current_volume > avg_volume * 1.5:
                signals.append("📈 Volume đột biến: Cần theo dõi")
            
            if not signals:
                signals.append("🟡 Không có tín hiệu rõ ràng")
            
            return f"🎯 TÍN HIỆU {symbol}:\n" + "\n".join(signals)
            
        except Exception as e:
            logger.error(f"Lỗi get trading signals {symbol}: {e}")
            return ""

    # ====================================================================
    # REALTIME VNDirect PRICES WITH TIMEOUT HANDLING
    # ====================================================================
    @st.cache_data(ttl=CACHE_TTL_REALTIME, show_spinner=False)
    def _get_realtime_prices(_self, symbols_tuple) -> str:
        """
        Fetch realtime prices, skip timeout error silently.
        Sử dụng BATCH FETCH để tối ưu hiệu suất.
        Cache 60s.
        """
        symbols = list(symbols_tuple)
        
        try:
            api = get_vndirect_api()
            
            # 1. Sử dụng get_multiple_stocks (Batch Fetch)
            # Giới hạn số lượng mã
            limited_symbols = symbols[:5] 
            
            # Gọi Batch API với timeout ngắn
            price_data = api.get_multiple_stocks(limited_symbols)
            
            # Định dạng kết quả
            results = []
            for symbol in limited_symbols:
                data = price_data.get(symbol)
                if data:
                    # Sử dụng format_stock_info từ VNDirect API client
                    results.append(api.format_stock_info(data))
            
            return "\n\n".join(results) if results else ""

        except (TimeoutError, ConnectionError, Exception) as e:
            # Im lặng bỏ qua lỗi timeout/API - không return error message
            logger.warning(f"Không thể lấy giá từ VNDirect (timeout/connection): {type(e).__name__}")
            return ""  # Trả về empty string thay vì error message

    # ====================================================================
    # PORTFOLIO CONTEXT
    # ====================================================================
    def get_portfolio_context(self, selected_stocks=None, optimization_result=None):
        parts = []
        if selected_stocks:
            parts.append(f"Stocks: {', '.join(selected_stocks)}")
        if optimization_result:
            parts.append(f"Results: {optimization_result}")
        return "\n".join(parts) if parts else None

    # ====================================================================
    # HISTORY MANAGEMENT
    # ====================================================================
    def _load_history_from_file(self):
        """Tải lịch sử từ file khi khởi động"""
        try:
            messages = self.history_manager.get_messages(self.session_id)
            if messages:
                self.chat_history = messages
                logger.info(f"✅ Đã khôi phục {len(messages)} tin nhắn từ session: {self.session_id}")
            else:
                logger.info(f"📂 Không có lịch sử cho session: {self.session_id}")
        except Exception as e:
            logger.error(f"⚠️ Lỗi khi tải lịch sử: {e}")
            self.chat_history = []
    
    def _save_history_to_file(self, metadata: Optional[Dict] = None):
        """Lưu lịch sử vào file sau mỗi câu hỏi"""
        try:
            # Chuyển đổi lịch sử chat (list of dict) sang định dạng cần lưu
            history_to_save = [
                {"role": "user", "content": h["user"]} if "user" in h else {"role": "model", "content": h["assistant"]} 
                for h in self.chat_history
            ]
            
            self.history_manager.save_history(
                messages=self.chat_history,
                session_id=self.session_id,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"⚠️ Lỗi khi lưu lịch sử: {e}")
    
    def clear_history(self):
        """Xóa lịch sử chat và file lưu trữ"""
        self.chat_history = []
        self.history_manager.clear_history(self.session_id)
        logger.info(f"🗑️ Đã xóa lịch sử session: {self.session_id}")
    
    def get_history_summary(self) -> Dict:
        """Lấy thông tin tóm tắt về lịch sử"""
        return {
            "session_id": self.session_id,
            "message_count": len(self.chat_history),
            "has_saved_file": self.history_manager.get_session_info(self.session_id) is not None
        }
    
    def export_history(self) -> str:
        """Xuất lịch sử sang text format"""
        return self.history_manager.export_to_text(self.session_id) or "Không có lịch sử"


# Quick questions
@st.cache_data(show_spinner=False)
def create_quick_question_buttons() -> List[str]:
    """Câu hỏi gợi ý với khả năng phân tích realtime"""
    return [
        "📊 Thị trường hôm nay thế nào?",
        "📈 Phân tích kỹ thuật VCB?",
        "🔥 So sánh VCB vs BID?",
        "📰 Tin tức HPG ảnh hưởng ra sao?",
        "💡 RSI VCB đang ở đâu?",
        "🎯 Top cổ phiếu hôm nay?",
    ]