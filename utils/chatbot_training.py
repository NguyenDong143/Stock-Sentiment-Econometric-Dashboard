"""
Chatbot Training Module - Tạo dataset và fine-tune chatbot
Cung cấp thông tin thị trường theo thời gian thực
"""

import pandas as pd
import json
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import streamlit as st
import logging

from utils.data_loader import load_price_data, load_sentiment_data, load_realtime_price_quote
from utils.vndirect_api import get_vndirect_api

logger = logging.getLogger(__name__)


# ================================================================
# TRAINING DATA GENERATOR - Tạo Q&A pairs từ dữ liệu thực
# ================================================================
class ChatbotTrainingDataGenerator:
    """
    Tạo training data cho chatbot từ dữ liệu lịch sử và realtime
    """
    
    def __init__(self, ticker_list: List[str] = None):
        self.ticker_list = ticker_list or ["VCB", "BID", "CTG", "TCB", "HPG", "VHM"]
        self.training_data = []
    
    def generate_price_analysis_qa(self, ticker: str) -> List[Dict]:
        """Tạo Q&A về phân tích giá"""
        qa_pairs = []
        
        try:
            # Lấy dữ liệu giá
            df = load_price_data(ticker)
            if df.empty or len(df) < 30:
                return []
            
            # Tính toán các chỉ số
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            # SMA
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else None
            
            # Volume
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Tạo Q&A pairs
            
            # 1. Giá hiện tại
            qa_pairs.append({
                "question": f"Giá {ticker} hôm nay bao nhiêu?",
                "answer": f"Giá {ticker} hiện tại là {current_price:,.0f} VNĐ, {'+' if change > 0 else ''}{change:,.0f} ({change_pct:+.2f}%). "
                         f"{'Tăng' if change > 0 else 'Giảm'} so với phiên trước.",
                "context": {
                    "ticker": ticker,
                    "price": current_price,
                    "change": change,
                    "change_pct": change_pct,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # 2. Xu hướng SMA
            if sma_20:
                trend = "tăng" if current_price > sma_20 else "giảm"
                qa_pairs.append({
                    "question": f"Xu hướng giá {ticker}?",
                    "answer": f"{ticker} đang {trend} (giá {current_price:,.0f} {'trên' if current_price > sma_20 else 'dưới'} SMA20: {sma_20:,.0f}). "
                             f"{'Tín hiệu tích cực' if current_price > sma_20 else 'Cần thận trọng'}.",
                    "context": {
                        "ticker": ticker,
                        "sma_20": sma_20,
                        "trend": trend
                    }
                })
            
            # 3. Golden Cross
            if sma_50:
                golden_cross = sma_20 > sma_50
                qa_pairs.append({
                    "question": f"{ticker} có golden cross không?",
                    "answer": f"{'Có' if golden_cross else 'Không'}. SMA(20)={sma_20:,.0f} {'>' if golden_cross else '<'} SMA(50)={sma_50:,.0f}. "
                             f"{'Tín hiệu xu hướng tăng trung hạn' if golden_cross else 'Chưa có tín hiệu xu hướng tăng mạnh'}.",
                    "context": {
                        "ticker": ticker,
                        "sma_20": sma_20,
                        "sma_50": sma_50,
                        "golden_cross": golden_cross
                    }
                })
            
            # 4. RSI Analysis
            if current_rsi:
                rsi_status = "quá mua" if current_rsi > 70 else "quá bán" if current_rsi < 30 else "trung tính"
                qa_pairs.append({
                    "question": f"RSI {ticker} bao nhiêu?",
                    "answer": f"RSI(14) của {ticker} là {current_rsi:.1f} - vùng {rsi_status}. "
                             f"{'Có thể điều chỉnh giảm' if current_rsi > 70 else 'Cơ hội mua vào' if current_rsi < 30 else 'Vùng an toàn'}.",
                    "context": {
                        "ticker": ticker,
                        "rsi": current_rsi,
                        "status": rsi_status
                    }
                })
            
            # 5. Volume Analysis
            volume_status = "mạnh" if volume_ratio > 1.5 else "yếu" if volume_ratio < 0.7 else "bình thường"
            qa_pairs.append({
                "question": f"Volume {ticker} như thế nào?",
                "answer": f"Volume giao dịch {ticker} {volume_status} ({volume_ratio:.1f}x trung bình 20 phiên). "
                         f"{'Thanh khoản tốt' if volume_ratio > 1.2 else 'Thanh khoản hạn chế'}.",
                "context": {
                    "ticker": ticker,
                    "volume_ratio": volume_ratio,
                    "status": volume_status
                }
            })
            
        except Exception as e:
            logger.error(f"Lỗi tạo Q&A giá cho {ticker}: {e}")
        
        return qa_pairs
    
    def generate_sentiment_qa(self, ticker: str) -> List[Dict]:
        """Tạo Q&A về phân tích cảm xúc tin tức"""
        qa_pairs = []
        
        try:
            # Lấy sentiment data
            df = load_sentiment_data(
                ticker=ticker,
                data_type="Content",
                time_period="After Scandal"
            )
            
            if df.empty or 'sentiment_label' not in df.columns:
                return []
            
            # Phân tích sentiment
            total = len(df)
            sentiment_counts = df['sentiment_label'].value_counts()
            
            positive = sentiment_counts.get(1, 0)
            negative = sentiment_counts.get(-1, 0)
            neutral = sentiment_counts.get(0, 0)
            
            # Tính tỷ lệ
            positive_pct = (positive / total) * 100
            negative_pct = (negative / total) * 100
            neutral_pct = (neutral / total) * 100
            
            # Xu hướng
            overall_sentiment = "tích cực" if positive > negative else "tiêu cực" if negative > positive else "trung tính"
            
            # Tạo Q&A
            qa_pairs.append({
                "question": f"Tin tức về {ticker} như thế nào?",
                "answer": f"Phân tích {total} bài viết về {ticker}: {positive_pct:.1f}% tích cực, {negative_pct:.1f}% tiêu cực, {neutral_pct:.1f}% trung tính. "
                         f"Xu hướng chung: {overall_sentiment}.",
                "context": {
                    "ticker": ticker,
                    "total": total,
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "sentiment": overall_sentiment
                }
            })
            
            qa_pairs.append({
                "question": f"Sentiment {ticker}?",
                "answer": f"Cảm xúc thị trường về {ticker} đang {overall_sentiment} "
                         f"({'Tích cực' if positive > negative * 1.5 else 'Thận trọng' if negative > positive else 'Ổn định'}).",
                "context": {
                    "ticker": ticker,
                    "sentiment": overall_sentiment
                }
            })
            
        except Exception as e:
            logger.error(f"Lỗi tạo Q&A sentiment cho {ticker}: {e}")
        
        return qa_pairs
    
    def generate_comparison_qa(self, ticker1: str, ticker2: str) -> List[Dict]:
        """Tạo Q&A so sánh 2 mã"""
        qa_pairs = []
        
        try:
            df1 = load_price_data(ticker1)
            df2 = load_price_data(ticker2)
            
            if df1.empty or df2.empty:
                return []
            
            # So sánh hiệu suất 30 ngày
            if len(df1) >= 30 and len(df2) >= 30:
                perf1 = ((df1['close'].iloc[-1] / df1['close'].iloc[-30]) - 1) * 100
                perf2 = ((df2['close'].iloc[-1] / df2['close'].iloc[-30]) - 1) * 100
                
                winner = ticker1 if perf1 > perf2 else ticker2
                
                qa_pairs.append({
                    "question": f"So sánh {ticker1} và {ticker2}?",
                    "answer": f"30 ngày qua: {ticker1} {perf1:+.2f}%, {ticker2} {perf2:+.2f}%. "
                             f"{winner} tốt hơn với chênh lệch {abs(perf1 - perf2):.2f}%.",
                    "context": {
                        "ticker1": ticker1,
                        "ticker2": ticker2,
                        "perf1": perf1,
                        "perf2": perf2,
                        "winner": winner
                    }
                })
            
        except Exception as e:
            logger.error(f"Lỗi tạo Q&A so sánh {ticker1} vs {ticker2}: {e}")
        
        return qa_pairs
    
    def generate_all_training_data(self) -> List[Dict]:
        """Tạo toàn bộ training data cho tất cả mã"""
        all_qa = []
        
        st.info("🤖 Đang tạo training data cho chatbot...")
        progress_bar = st.progress(0)
        
        total_tickers = len(self.ticker_list)
        
        for idx, ticker in enumerate(self.ticker_list):
            # Price analysis
            price_qa = self.generate_price_analysis_qa(ticker)
            all_qa.extend(price_qa)
            
            # Sentiment analysis
            sentiment_qa = self.generate_sentiment_qa(ticker)
            all_qa.extend(sentiment_qa)
            
            # Progress
            progress_bar.progress((idx + 1) / total_tickers)
        
        # Comparison pairs
        for i in range(len(self.ticker_list) - 1):
            comp_qa = self.generate_comparison_qa(
                self.ticker_list[i], 
                self.ticker_list[i + 1]
            )
            all_qa.extend(comp_qa)
        
        progress_bar.empty()
        st.success(f"✅ Đã tạo {len(all_qa)} Q&A pairs!")
        
        self.training_data = all_qa
        return all_qa
    
    def save_training_data(self, filepath: str = "data/chatbot_training.json"):
        """Lưu training data ra file JSON"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Đã lưu {len(self.training_data)} training samples vào {filepath}")
        return filepath
    
    def load_training_data(self, filepath: str = "data/chatbot_training.json") -> List[Dict]:
        """Load training data từ file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
            logger.info(f"Đã load {len(self.training_data)} training samples")
            return self.training_data
        except Exception as e:
            logger.error(f"Lỗi load training data: {e}")
            return []


# ================================================================
# MARKET CONTEXT PROVIDER - Cung cấp context realtime cho chatbot
# ================================================================
class MarketContextProvider:
    """
    Cung cấp context thị trường realtime cho chatbot
    """
    
    @staticmethod
    @st.cache_data(ttl=60, show_spinner=False)
    def get_market_overview() -> str:
        """Lấy tổng quan thị trường"""
        try:
            api = get_vndirect_api()
            overview = api.get_market_overview()
            
            if not overview:
                return ""
            
            vn_data = overview.get('vnindex')
            hnx_data = overview.get('hnxindex')
            
            result = "📊 TỔNG QUAN THỊ TRƯỜNG:\n"
            
            if vn_data:
                result += f"- VNINDEX: {vn_data['price']:,.2f} ({vn_data['change']:+,.2f} | {vn_data['change_percent']:+.2f}%)\n"
            
            if hnx_data:
                result += f"- HNXINDEX: {hnx_data['price']:,.2f} ({hnx_data['change']:+,.2f} | {hnx_data['change_percent']:+.2f}%)\n"
            
            result += f"- Cập nhật: {overview['time']}"
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi get market overview: {e}")
            return ""
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def get_sector_performance(sector_tickers: List[str]) -> str:
        """Lấy hiệu suất ngành"""
        try:
            performances = []
            
            for ticker in sector_tickers:
                df = load_price_data(ticker)
                if not df.empty and len(df) >= 5:
                    perf_5d = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100
                    performances.append({
                        "ticker": ticker,
                        "performance": perf_5d
                    })
            
            if not performances:
                return ""
            
            # Sắp xếp theo performance
            performances.sort(key=lambda x: x['performance'], reverse=True)
            
            result = "🏆 HIỆU SUẤT 5 NGÀY:\n"
            for item in performances[:5]:  # Top 5
                result += f"- {item['ticker']}: {item['performance']:+.2f}%\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi get sector performance: {e}")
            return ""
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def get_trading_signals(ticker: str) -> str:
        """Lấy tín hiệu giao dịch"""
        try:
            df = load_price_data(ticker)
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
            
            if not signals:
                signals.append("🟡 Không có tín hiệu rõ ràng")
            
            return f"🎯 TÍN HIỆU {ticker}:\n" + "\n".join(signals)
            
        except Exception as e:
            logger.error(f"Lỗi get trading signals {ticker}: {e}")
            return ""


# ================================================================
# USAGE EXAMPLE
# ================================================================
if __name__ == "__main__":
    # Test training data generator
    generator = ChatbotTrainingDataGenerator(["VCB", "BID", "CTG"])
    training_data = generator.generate_all_training_data()
    
    print(f"Generated {len(training_data)} training samples")
    print("\nSample:")
    print(json.dumps(training_data[0], ensure_ascii=False, indent=2))
    
    # Save
    generator.save_training_data()
