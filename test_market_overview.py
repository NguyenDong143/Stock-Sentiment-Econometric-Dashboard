"""
Test market overview query
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("TEST: Market Overview Query")
print("="*60)

from models.chatbot_services import PortfolioChatbot
from config.settings import GEMINI_API_KEY

try:
    print("\n1. Khởi tạo chatbot...")
    bot = PortfolioChatbot(
        GEMINI_API_KEY,
        session_id="market_test",
        auto_load=False
    )
    print("✅ OK")
    
    print("\n2. Test market overview detection...")
    test_queries = [
        "Thị trường hôm nay thế nào?",
        "VNINDEX hôm nay?",
        "Tình hình thị trường?",
        "Giao dịch hôm nay ra sao?"
    ]
    
    for query in test_queries:
        is_market = bot._is_market_overview_query(query)
        print(f"   {'✅' if is_market else '❌'} '{query}' → {is_market}")
    
    print("\n3. Test response generation...")
    test_question = "Thị trường hôm nay thế nào?"
    print(f"   Question: {test_question}")
    print(f"   Generating response...\n")
    print("   " + "-"*50)
    
    response = bot.generate_response(test_question)
    print(response)
    
    print("   " + "-"*50)
    print("\n✅ Test completed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
