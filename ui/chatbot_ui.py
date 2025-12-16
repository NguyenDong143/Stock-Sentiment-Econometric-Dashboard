import streamlit as st
from models.chatbot_services import PortfolioChatbot, create_quick_question_buttons
import time


# ============================================================
# CSS STYLING
# ============================================================
def inject_chatbot_css():
    """Inject CSS cho chatbot UI"""
    st.markdown("""
    <style>
        /* ===== FLOATING BUTTON STYLE ===== */
        div[data-testid="stButton"] button[kind="primary"] {
            position: fixed !important;
            bottom: 20px !important;
            right: 20px !important;
            z-index: 9999 !important;
            width: 60px !important;
            height: 60px !important;
            border-radius: 50% !important;
            font-size: 28px !important;
            padding: 0 !important;
            box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4) !important;
            transition: all 0.3s ease !important;
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
            border: none !important;
        }
        
        div[data-testid="stButton"] button[kind="primary"]:hover {
            transform: scale(1.1) !important;
            box-shadow: 0 6px 20px rgba(34, 197, 94, 0.6) !important;
        }
        
        /* ===== CHAT CONTAINER STYLE ===== */
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 8px;
        }
        
        /* ===== QUICK QUESTIONS STYLE ===== */
        div[data-testid="stButton"] button[kind="secondary"] {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
            border: 1px solid #38bdf8 !important;
            color: #0c4a6e !important;
            transition: all 0.2s ease !important;
            font-size: 13px !important;
            padding: 8px 12px !important;
        }
        
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
            color: white !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(56, 189, 248, 0.3) !important;
        }
        
        /* ===== CHAT MESSAGE STYLE ===== */
        div[data-testid="stChatMessage"] {
            border-radius: 12px !important;
            padding: 10px !important;
            margin-bottom: 6px !important;
        }
        
        div[data-testid="stChatMessage"][data-testid*="user"] {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        }
        
        div[data-testid="stChatMessage"][data-testid*="assistant"] {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important;
        }
        
        /* ===== TYPING INDICATOR ===== */
        .typing-indicator {
            display: inline-block;
            animation: blink 1.4s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        /* ===== STREAMING TEXT ANIMATION ===== */
        .streaming-text {
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# KHỞI TẠO CHATBOT SESSION
# ============================================================
def initialize_chatbot_session():
    """Khởi tạo chatbot và session state - SINGLE KEY SUPPORT"""
    # Sử dụng session_id cố định
    session_id = "global_chat"
    
    # CHỈ khởi tạo chatbot khi popup đã mở VÀ chưa có chatbot
    if 'chatbot' not in st.session_state:
        if st.session_state.get('show_chatbot_popup', False):
            try:
                from config.settings import GEMINI_API_KEY
                
                if not GEMINI_API_KEY:
                    st.session_state.chatbot = None
                    st.session_state.chatbot_error = "❌ Chưa cấu hình GEMINI_API_KEY"
                else:
                    # Tạo chatbot với single key
                    st.session_state.chatbot = PortfolioChatbot(
                        GEMINI_API_KEY,
                        session_id=session_id,
                        auto_load=True
                    )
                    st.session_state.chatbot_error = None
                    print("✅ Chatbot khởi tạo thành công")
                        
            except Exception as e:
                st.session_state.chatbot = None
                st.session_state.chatbot_error = f"⚠️ Lỗi khởi tạo chatbot: {str(e)}"
                print(f"❌ CHI TIẾT LỖI CHATBOT: {e}")
                import traceback
                traceback.print_exc()

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": """👋 Xin chào! Tôi là **AI Assistant phân tích cổ phiếu thời gian thực**.

🎯 **Khả năng của tôi:**

📊 **Phân tích Realtime**
• Giá hiện tại, thay đổi, xu hướng
• SMA(20/50), Golden Cross
• RSI, Volume, tín hiệu kỹ thuật

📰 **Phân tích Tin tức**
• Sentiment từ PhoBERT
• Tác động tin tức lên giá

🔍 **So sánh & Tư vấn**
• So sánh mã cổ phiếu
• Phân tích rủi ro/cơ hội

Hỏi tôi bất kỳ điều gì về thị trường! 🚀"""
        }]

    if 'show_quick_questions' not in st.session_state:
        st.session_state.show_quick_questions = True
    
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0


# ============================================================
# XỬ LÝ TIN NHẮN VỚI STREAMING
# ============================================================
def handle_user_message_stream(user_message):
    """
    Xử lý tin nhắn với streaming response (real-time typing effect)
    """
    # Thêm user message
    st.session_state.chat_messages.append({
        "role": "user", 
        "content": user_message
    })
    
    # Hiển thị user message ngay lập tức
    with st.chat_message("user"):
        st.markdown(user_message)
    
    # Tạo assistant message với streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show typing indicator
        message_placeholder.markdown("🤖 <span class='typing-indicator'>●●●</span>", unsafe_allow_html=True)
        
        # Stream response chunks
        try:
            for chunk in st.session_state.chatbot.generate_response_stream(user_message, context=None):
                full_response += chunk
                # Update với cursor để thấy effect đang gõ
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.001)  # Reduced delay cho smoother streaming
            
            # Remove cursor khi hoàn thành
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"⚠️ Lỗi: {str(e)}"
            message_placeholder.markdown(full_response)
    
    # Lưu vào history
    st.session_state.chat_messages.append({
        "role": "assistant", 
        "content": full_response
    })


# ============================================================
# RENDER QUICK QUESTIONS
# ============================================================
def render_quick_questions():
    """Hiển thị câu hỏi gợi ý compact"""
    st.markdown("---")
    st.markdown("#### 💡 Câu hỏi gợi ý:")
    
    questions = create_quick_question_buttons()
    
    # Hiển thị 2 cột compact
    for row in range(0, min(len(questions), 6), 2):
        cols = st.columns(2)
        for col_idx, q_idx in enumerate(range(row, min(row + 2, len(questions)))):
            with cols[col_idx]:
                st.button(
                    questions[q_idx], 
                    key=f"quick_q_{q_idx}",
                    use_container_width=True,
                    type="secondary",
                    on_click=lambda q=questions[q_idx]: handle_quick_question_click(q)
                )


def handle_quick_question_click(question):
    """Callback khi click quick question"""
    st.session_state.pending_question = question
    st.session_state.show_quick_questions = False


# ============================================================
# RENDER CHAT HISTORY
# ============================================================
def render_chat_history():
    """Hiển thị chat history"""
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ============================================================
# EXPORT CHAT HISTORY
# ============================================================
def export_chat_history():
    """Xuất chat nhanh"""
    from datetime import datetime
    
    chat_text = "=" * 50 + "\n"
    chat_text += "CHAT HISTORY - AI ASSISTANT\n"
    chat_text += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    chat_text += "=" * 50 + "\n\n"
    
    for msg in st.session_state.chat_messages:
        role = "YOU" if msg["role"] == "user" else "AI"
        chat_text += f"{role}: {msg['content']}\n\n"
    
    return chat_text


# ============================================================
# CALLBACK FUNCTIONS
# ============================================================
def clear_chat_callback():
    """Xóa chat và file lưu trữ"""
    st.session_state.chat_messages = [st.session_state.chat_messages[0]]
    st.session_state.show_quick_questions = True
    if st.session_state.chatbot:
        st.session_state.chatbot.clear_history()
        st.success("🗑️ Đã xóa lịch sử chat!")


def show_quick_questions_callback():
    """Show gợi ý"""
    st.session_state.show_quick_questions = True


# ============================================================
# FLOATING BUTTON
# ============================================================
def render_floating_button():
    """Render floating button - GIỮ STATE ỔN ĐỊNH"""
    inject_chatbot_css()
    
    def toggle_chatbot():
        st.session_state.show_chatbot_popup = not st.session_state.get('show_chatbot_popup', False)
    
    # Khởi tạo state - mặc định ĐÓNG
    if 'show_chatbot_popup' not in st.session_state:
        st.session_state.show_chatbot_popup = False
    
    # GIỮ CHATBOT STATE - KHÔNG tự động đóng khi sidebar thay đổi
    # Điều này tránh việc reset page và mất thời gian khởi tạo
    
    st.button(
        "🤖", 
        key="chatbot_floating_btn", 
        help="AI Chat",
        type="primary",
        on_click=toggle_chatbot
    )


# ============================================================
# DIALOG CONTENT
# ============================================================
def render_dialog_content():
    """Render dialog với thông tin chatbot realtime"""
    # Force init chatbot khi dialog mở
    if 'chatbot' not in st.session_state:
        initialize_chatbot_session()
    
    # Header với nút đóng
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("### 💬 AI Assistant - Market Analysis")
    with col2:
        def close_chatbot():
            st.session_state.show_chatbot_popup = False
        st.button("✖", key="close_chatbot_btn", type="secondary", on_click=close_chatbot)
    
    # Thông tin khả năng
    with st.expander("ℹ️ Thông tin Chatbot", expanded=False):
        st.markdown("""
        **🎯 Chatbot được huấn luyện với:**
        
        ✅ **Dữ liệu Realtime:** VNDirect API (giá, volume)  
        ✅ **Phân tích Kỹ thuật:** RSI, SMA, Golden Cross, Volume  
        ✅ **Sentiment Analysis:** PhoBERT trên tin tức VN  
        ✅ **So sánh & Tư vấn:** Performance, tín hiệu trading
        
        **💡 Ví dụ câu hỏi:**
        - "Phân tích kỹ thuật VCB?"
        - "RSI VCB bao nhiêu?"
        - "So sánh VCB với BID?"
        - "Tin tức VCB ảnh hưởng thế nào?"
        """)
    
    st.markdown("---")
    
    # Check lỗi chatbot
    if st.session_state.get('chatbot') is None:
        st.error(st.session_state.get('chatbot_error', '⚠️ Lỗi chatbot'))
        with st.expander("💡 Hướng dẫn"):
            st.code('GEMINI_API_KEY = "your-key"\n# Get key: https://makersuite.google.com/app/apikey')
        return
    
    # Render chat interface
    render_chat_history()
    
    # Chat input
    user_input = st.chat_input("💬 Nhập câu hỏi...", key="chat_input_dialog")
    
    if user_input and user_input.strip():
        handle_user_message_stream(user_input.strip())
        # Không cần st.rerun() - message đã hiển thị trong handle_user_message_stream


# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================
def render():
    """Render giao diện chính với streaming support"""
    inject_chatbot_css()
    initialize_chatbot_session()
    render_chat_interface()


def render_chat_interface():
    """Render chat interface (dùng chung cho dialog và standalone)"""
    # Force init if needed
    if 'chatbot' not in st.session_state:
        initialize_chatbot_session()
    
    # Check lỗi
    if st.session_state.get('chatbot') is None:
        st.error(st.session_state.get('chatbot_error', '⚠️ Lỗi chatbot'))
        with st.expander("💡 Hướng dẫn"):
            st.code('GEMINI_API_KEY = "your-key"\n# Get key: https://makersuite.google.com/app/apikey')
        return

    # Chat history
    render_chat_history()
    
    # Xử lý pending question từ quick buttons
    if 'pending_question' in st.session_state and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        handle_user_message_stream(question)

    # Quick questions
    if len(st.session_state.chat_messages) <= 1 and st.session_state.show_quick_questions:
        render_quick_questions()

    # Chat input
    user_input = st.chat_input(
        "💬 Nhập câu hỏi...",
        key=f"chat_input_{st.session_state.chat_input_key}"
    )
    
    if user_input and user_input.strip():
        handle_user_message_stream(user_input.strip())
        st.session_state.show_quick_questions = False
        st.session_state.chat_input_key += 1
        st.rerun()  # Rerun để show message mới

    # Session Info (compact)
    if st.session_state.chatbot:
        summary = st.session_state.chatbot.get_history_summary()
        st.caption(f"💾 Session: `{summary['session_id']}` | Tin nhắn: {summary['message_count']} | Đã lưu: {'✅' if summary['has_saved_file'] else '❌'}")
    
    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.button("🗑️ Xóa", use_container_width=True, on_click=clear_chat_callback)
    
    with col2:
        st.button("💡 Gợi ý", use_container_width=True, on_click=show_quick_questions_callback)
    
    with col3:
        chat_text = export_chat_history()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "💾 Xuất",
            chat_text,
            f"chat_export_{timestamp}.txt",
            "text/plain",
            use_container_width=True
        )


# ============================================================
# SHOW POPUP
# ============================================================
@st.dialog("🤖 AI Assistant", width="large")
def chatbot_dialog():
    """Dialog content"""
    render_dialog_content()

def show_popup_dialog():
    """Show popup - chỉ hiển thị khi user click button"""
    # Chỉ gọi dialog khi popup được bật rõ ràng
    if st.session_state.get('show_chatbot_popup', False):
        try:
            chatbot_dialog()
        except Exception as e:
            import logging
            logging.error(f"Dialog error: {e}")
            st.session_state.show_chatbot_popup = False
