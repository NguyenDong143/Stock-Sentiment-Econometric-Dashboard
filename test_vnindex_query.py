"""
Test VNINDEX query with improved chatbot
"""
import os
import sys

# Thiết lập API key (bạn cần thay bằng key thực)
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

from models.chatbot_services import PortfolioChatbot

def test_vnindex_queries():
    """Test các câu hỏi về VNINDEX"""
    
    print("=" * 60)
    print("TEST CHATBOT - VNINDEX QUERIES")
    print("=" * 60)
    
    # Khởi tạo chatbot
    chatbot = PortfolioChatbot(
        api_key=os.getenv("GEMINI_API_KEY"),
        session_id="vnindex_test",
        auto_load=False
    )
    
    # Các câu hỏi test
    test_queries = [
        "VNINDEX hôm nay?",
        "Thị trường thế nào?",
        "VN-INDEX bây giờ",
        "Chỉ số VN",
        "Tổng quan thị trường",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Câu hỏi {i}: {query}")
        print("-" * 60)
        
        try:
            # Test streaming response
            response_chunks = []
            for chunk in chatbot.generate_response_stream(query):
                response_chunks.append(chunk)
                print(chunk, end="", flush=True)
            
            print("\n" + "-" * 60)
            print(f"✅ Response length: {len(''.join(response_chunks))} chars")
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    # Kiểm tra API key
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "YOUR_API_KEY_HERE":
        print("⚠️ Cảnh báo: Chưa thiết lập GEMINI_API_KEY")
        print("Vui lòng sửa file này hoặc set environment variable:")
        print('  export GEMINI_API_KEY="your-key-here"')
        print("\nTiếp tục test với mock...")
    
    test_vnindex_queries()
