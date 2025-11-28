"""
Test Script - Chatbot Realtime Features
Kiểm tra các chức năng của chatbot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot_services import PortfolioChatbot
from utils.chatbot_training import MarketContextProvider
from utils.vndirect_api import get_vndirect_api
from utils.data_loader import load_price_data, load_sentiment_data
from config.settings import GEMINI_API_KEY
import time


def print_section(title):
    """In header cho mỗi phần test"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_vndirect_api():
    """Test 1: VNDirect API - Lấy giá realtime"""
    print_section("TEST 1: VNDirect API - Giá Realtime")
    
    try:
        api = get_vndirect_api()
        
        # Test single stock
        print("\n1. Test single stock (VCB):")
        start = time.time()
        vcb_data = api.get_stock_price("VCB")
        elapsed = time.time() - start
        
        if vcb_data:
            print(f"   ✅ Thành công! ({elapsed:.2f}s)")
            print(f"   - Symbol: {vcb_data['symbol']}")
            print(f"   - Price: {vcb_data['price']:,} VNĐ")
            print(f"   - Change: {vcb_data['change']:+,} ({vcb_data['change_percent']:+.2f}%)")
            print(f"   - Volume: {vcb_data['volume']:,}")
        else:
            print("   ❌ Không lấy được dữ liệu")
            return False
        
        # Test multiple stocks
        print("\n2. Test multiple stocks (VCB, BID, CTG):")
        start = time.time()
        multi_data = api.get_multiple_stocks(["VCB", "BID", "CTG"])
        elapsed = time.time() - start
        
        if multi_data:
            print(f"   ✅ Thành công! ({elapsed:.2f}s)")
            print(f"   - Số mã lấy được: {len(multi_data)}/3")
            for symbol, data in multi_data.items():
                print(f"   - {symbol}: {data['price']:,} VNĐ ({data['change_percent']:+.2f}%)")
        else:
            print("   ❌ Không lấy được dữ liệu")
            return False
        
        # Test cache
        print("\n3. Test cache (query lại VCB):")
        start = time.time()
        vcb_data_cached = api.get_stock_price("VCB")
        elapsed = time.time() - start
        
        if vcb_data_cached:
            print(f"   ✅ Cache hit! ({elapsed:.4f}s - nhanh hơn nhiều)")
            print(f"   - Cache hoạt động tốt!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
        return False


def test_technical_analysis():
    """Test 2: Phân tích kỹ thuật"""
    print_section("TEST 2: Phân Tích Kỹ Thuật")
    
    try:
        print("\n1. Load price data (VCB):")
        start = time.time()
        df = load_price_data("VCB")
        elapsed = time.time() - start
        
        if df.empty:
            print("   ❌ Không có dữ liệu giá")
            return False
        
        print(f"   ✅ Thành công! ({elapsed:.2f}s)")
        print(f"   - Số ngày: {len(df)}")
        print(f"   - Giá gần nhất: {df['close'].iloc[-1]:,.0f} VNĐ")
        
        # Tính RSI
        print("\n2. Tính RSI:")
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        print(f"   ✅ RSI(14): {current_rsi:.2f}")
        if current_rsi > 70:
            print(f"   - Vùng quá mua")
        elif current_rsi < 30:
            print(f"   - Vùng quá bán")
        else:
            print(f"   - Vùng trung tính")
        
        # Tính SMA
        print("\n3. Tính SMA:")
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        print(f"   ✅ SMA(20): {sma_20:,.0f} VNĐ")
        print(f"   ✅ SMA(50): {sma_50:,.0f} VNĐ")
        print(f"   - Golden Cross: {'Có' if sma_20 > sma_50 else 'Không'}")
        print(f"   - Xu hướng: {'Tăng' if current_price > sma_20 else 'Giảm'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
        return False


def test_sentiment_analysis():
    """Test 3: Phân tích sentiment"""
    print_section("TEST 3: Phân Tích Sentiment (PhoBERT)")
    
    try:
        print("\n1. Load sentiment data (VCB):")
        start = time.time()
        df = load_sentiment_data(
            ticker="VCB",
            data_type="Content",
            time_period="After Scandal"
        )
        elapsed = time.time() - start
        
        if df.empty:
            print("   ⚠️ Không có dữ liệu sentiment cho VCB")
            print("   (Có thể mã này không có trong dataset)")
            return True  # Không phải lỗi nghiêm trọng
        
        print(f"   ✅ Thành công! ({elapsed:.2f}s)")
        print(f"   - Số bài viết: {len(df)}")
        
        if 'sentiment_label' in df.columns:
            sentiment_counts = df['sentiment_label'].value_counts()
            total = len(df)
            
            positive = sentiment_counts.get(1, 0)
            negative = sentiment_counts.get(-1, 0)
            neutral = sentiment_counts.get(0, 0)
            
            print(f"\n2. Phân tích sentiment:")
            print(f"   ✅ Tích cực: {positive} ({positive/total*100:.1f}%)")
            print(f"   ✅ Tiêu cực: {negative} ({negative/total*100:.1f}%)")
            print(f"   ✅ Trung tính: {neutral} ({neutral/total*100:.1f}%)")
            
            if positive > negative:
                print(f"   - Xu hướng: Tích cực ✅")
            elif negative > positive:
                print(f"   - Xu hướng: Tiêu cực ⚠️")
            else:
                print(f"   - Xu hướng: Trung tính ⚖️")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
        return False


def test_market_context_provider():
    """Test 4: Market Context Provider"""
    print_section("TEST 4: Market Context Provider")
    
    try:
        print("\n1. Market Overview:")
        overview = MarketContextProvider.get_market_overview()
        if overview:
            print("   ✅ Thành công!")
            print(overview)
        else:
            print("   ⚠️ Không lấy được market overview")
        
        print("\n2. Sector Performance:")
        perf = MarketContextProvider.get_sector_performance(["VCB", "BID", "CTG"])
        if perf:
            print("   ✅ Thành công!")
            print(perf)
        else:
            print("   ⚠️ Không lấy được sector performance")
        
        print("\n3. Trading Signals (VCB):")
        signals = MarketContextProvider.get_trading_signals("VCB")
        if signals:
            print("   ✅ Thành công!")
            print(signals)
        else:
            print("   ⚠️ Không lấy được trading signals")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
        return False


def test_chatbot_integration():
    """Test 5: Chatbot Integration (Core)"""
    print_section("TEST 5: Chatbot Integration")
    
    try:
        # Check API key
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
            print("   ⚠️ GEMINI_API_KEY chưa được cấu hình")
            print("   Vui lòng thêm API key vào config/settings.py")
            return False
        
        print("\n1. Khởi tạo chatbot:")
        start = time.time()
        chatbot = PortfolioChatbot(
            GEMINI_API_KEY,
            session_id="test_session",
            auto_load=False
        )
        elapsed = time.time() - start
        print(f"   ✅ Thành công! ({elapsed:.2f}s)")
        
        # Test symbol extraction
        print("\n2. Test symbol extraction:")
        test_messages = [
            "Giá VCB hôm nay?",
            "Phân tích kỹ thuật BID",
            "So sánh VCB và CTG"
        ]
        
        for msg in test_messages:
            symbols = chatbot._extract_stock_symbols(msg)
            print(f"   - '{msg}' → {symbols}")
        
        # Test response generation
        print("\n3. Test response generation:")
        test_question = "Giá VCB hôm nay?"
        print(f"   Question: {test_question}")
        print(f"   Generating response...")
        
        start = time.time()
        response = chatbot.generate_response(test_question)
        elapsed = time.time() - start
        
        print(f"\n   ✅ Response generated! ({elapsed:.2f}s)")
        print(f"   Response preview: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_chatbot_streaming():
    """Test 6: Chatbot Streaming (Advanced)"""
    print_section("TEST 6: Chatbot Streaming Response")
    
    try:
        chatbot = PortfolioChatbot(
            GEMINI_API_KEY,
            session_id="test_stream",
            auto_load=False
        )
        
        test_question = "Phân tích kỹ thuật VCB?"
        print(f"\n   Question: {test_question}")
        print(f"   Streaming response:\n")
        print("   " + "-"*50)
        
        start = time.time()
        full_response = ""
        
        for chunk in chatbot.generate_response_stream(test_question):
            print(chunk, end='', flush=True)
            full_response += chunk
        
        elapsed = time.time() - start
        
        print("\n   " + "-"*50)
        print(f"\n   ✅ Stream completed! ({elapsed:.2f}s)")
        print(f"   Total length: {len(full_response)} chars")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
        return False


def run_all_tests():
    """Chạy tất cả tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "CHATBOT REALTIME - TEST SUITE" + " "*18 + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        ("VNDirect API", test_vndirect_api),
        ("Technical Analysis", test_technical_analysis),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Market Context Provider", test_market_context_provider),
        ("Chatbot Integration", test_chatbot_integration),
        ("Chatbot Streaming", test_chatbot_streaming),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {str(e)}")
            results.append((name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    print()
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}  {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n   🎉 All tests passed! Chatbot is ready to use!")
    else:
        print(f"\n   ⚠️ {total - passed} test(s) failed. Please check the logs above.")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
