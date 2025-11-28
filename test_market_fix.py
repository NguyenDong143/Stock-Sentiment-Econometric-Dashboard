"""Test chatbot với câu hỏi market overview sau khi fix system prompt"""
import os
import sys

# Set API key
os.environ['GEMINI_API_KEY'] = 'AIzaSyDLW9eJV2V9_ZCaIPPsb8KlzpINdZM7Hc4'

# Import chatbot services
from models.chatbot_services import PortfolioChatbot

def test_market_query():
    print("=== TEST MARKET OVERVIEW RESPONSE ===\n")
    
    # Initialize chatbot
    print("1. Khởi tạo chatbot...")
    api_key = os.environ.get('GEMINI_API_KEY')
    chatbot = PortfolioChatbot(api_key=api_key)
    print("   ✅ Khởi tạo thành công\n")
    
    # Test query
    query = "thị trường hôm nay thế nào"
    print(f"2. Gửi câu hỏi: '{query}'\n")
    
    # Get response
    print("3. Nhận phản hồi:")
    print("-" * 60)
    
    full_response = ""
    for chunk in chatbot.generate_response_stream(query, [], "test_session"):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n" + "-" * 60)
    
    # Analyze response
    print("\n4. Phân tích kết quả:")
    
    # Check if response is generic AI disclaimer
    generic_phrases = [
        "Tôi là AI",
        "không thể cung cấp thông tin thị trường theo thời gian thực",
        "I'm an AI",
        "I cannot provide real-time"
    ]
    
    is_generic = any(phrase.lower() in full_response.lower() for phrase in generic_phrases)
    
    if is_generic:
        print("   ❌ FAILED: Bot vẫn trả lời generic")
        print(f"   Response chứa disclaimer về AI")
    else:
        print("   ✅ SUCCESS: Bot đã phân tích market data")
        
        # Check for market-related keywords
        market_keywords = ["VNINDEX", "thị trường", "tăng", "giảm", "điểm", "%", "blue chip"]
        found_keywords = [kw for kw in market_keywords if kw.lower() in full_response.lower()]
        
        if found_keywords:
            print(f"   Found keywords: {', '.join(found_keywords)}")
        else:
            print("   ⚠️  WARNING: Response không chứa keywords thị trường")
    
    return full_response

if __name__ == "__main__":
    try:
        response = test_market_query()
        print("\n=== TEST COMPLETED ===")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
