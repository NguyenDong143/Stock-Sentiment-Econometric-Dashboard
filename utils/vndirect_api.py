"""
VNDirect REALTIME API Client – Final PRO Version (2025)
Optimized for Streamlit Dashboard & Chatbot
"""

import time
import requests
from typing import Dict, List, Optional
from datetime import datetime


# ================================================================
# GLOBAL CACHE (5 seconds)
# SỬA LỖI: Thêm cơ chế dọn dẹp (cleanup) mục hết hạn
# ================================================================
_CACHE = {}
_CACHE_EXPIRY = {}


def cache_get(key: str):
    """Get item from cache if not expired. Clean up expired keys."""
    if key in _CACHE:
        # Kiểm tra hết hạn
        if time.time() < _CACHE_EXPIRY[key]:
            return _CACHE[key]
        
        # Xóa mục hết hạn khỏi bộ nhớ để tránh rò rỉ
        del _CACHE[key]
        del _CACHE_EXPIRY[key]
        
    return None


def cache_set(key: str, value, ttl=30):
    """Store item in cache (default 30s)"""
    _CACHE[key] = value
    _CACHE_EXPIRY[key] = time.time() + ttl


# ================================================================
# MAIN CLASS (Giữ nguyên)
# ================================================================
class VNDirectAPI:
    """
    Realtime stock quote client for VNDirect
    Endpoint: https://finfo-api.vndirect.com.vn/v3/stocks/quotes
    """

    BASE_URL = "https://finfo-api.vndirect.com.vn"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

    # ============================================================
    # FETCH SINGLE STOCK
    # ============================================================
    def get_stock_price(self, symbol: str, retry: int = 2) -> Optional[Dict]:
        symbol = symbol.upper()

        # 1) Check cache
        cached = cache_get(f"price_{symbol}")
        if cached:
            return cached

        for attempt in range(retry):
            try:
                url = f"{self.BASE_URL}/v3/stocks/quotes"
                params = {
                    "q": f"code:{symbol}",
                    "size": 1
                }

                r = self.session.get(url, params=params, timeout=8)
                r.raise_for_status()
                data = r.json()

                # CASE 1: Symbol does not exist - im lặng bỏ qua
                if not data.get("data"):
                    return None

                d = data["data"][0]

                result = {
                    "symbol": d.get("code"),
                    "price": d.get("lastPrice"),
                    "change": d.get("change"),
                    "change_percent": d.get("pctChange"),
                    "volume": d.get("matchedVolume"),
                    "high": d.get("high"),
                    "low": d.get("low"),
                    "floor_price": d.get("floor"),
                    "ceiling_price": d.get("ceiling"),
                    "ref_price": d.get("reference"),
                    "time": datetime.now().strftime("%H:%M:%S")
                }

                # Store to cache
                cache_set(f"price_{symbol}", result)

                return result

            except Exception as e:
                # Retry sau 0.5s nếu thất bại
                if attempt < retry - 1:
                    time.sleep(0.5)
                continue

        return None

    # ============================================================
    # FETCH MULTIPLE STOCKS (BATCH + CACHE)
    # ============================================================
    def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch many symbols at once using batch query with retry"""
        symbols = [s.upper() for s in symbols]

        results = {}

        # 1) Check cache first
        remaining = []
        for sym in symbols:
            cached = cache_get(f"price_{sym}")
            if cached:
                results[sym] = cached
            else:
                remaining.append(sym)

        if not remaining:
            return results

        # 2) Batch fetch với retry
        for attempt in range(2):
            try:
                query = " OR ".join([f"code:{s}" for s in remaining])
                url = f"{self.BASE_URL}/v3/stocks/quotes"
                params = {
                    "q": query,
                    "size": len(remaining)
                }

                r = self.session.get(url, params=params, timeout=8)
                r.raise_for_status()
                data = r.json().get("data", [])

                for d in data:
                    sym = d["code"]
                    result = {
                        "symbol": sym,
                        "price": d.get("lastPrice"),
                        "change": d.get("change"),
                        "change_percent": d.get("pctChange"),
                        "volume": d.get("matchedVolume"),
                        "high": d.get("high"),
                        "low": d.get("low"),
                        "floor_price": d.get("floor"),
                        "ceiling_price": d.get("ceiling"),
                        "ref_price": d.get("reference"),
                        "time": datetime.now().strftime("%H:%M:%S")
                    }
                    results[sym] = result
                    cache_set(f"price_{sym}", result)
                
                break  # Thành công, thoát vòng lặp
                
            except Exception as e:
                # Retry sau 0.5s nếu thất bại
                if attempt < 1:
                    time.sleep(0.5)
                continue

        return results

    # ============================================================
    # COMPANY INFO
    # ============================================================
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Lấy thông tin cơ bản của công ty"""
        symbol = symbol.upper()
        
        # Check cache
        cached = cache_get(f"company_{symbol}")
        if cached:
            return cached
        
        try:
            # VNDirect API cho thông tin công ty
            url = f"{self.BASE_URL}/v4/stocks"
            params = {
                "q": f"code:{symbol}",
                "size": 1
            }
            
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            if not data.get("data"):
                return None
            
            d = data["data"][0]
            
            result = {
                "symbol": d.get("code"),
                "company_name": d.get("companyName"),
                "company_name_eng": d.get("companyNameEng"),
                "short_name": d.get("shortName"),
                "exchange": d.get("exchange"),
                "industry": d.get("icbName3"),  # Ngành cấp 3
                "sector": d.get("icbName2"),     # Ngành cấp 2
                "website": d.get("website"),
                "established_date": d.get("establishedDate"),
                "listed_date": d.get("issueDate"),
            }
            
            # Cache 1 giờ (thông tin công ty ít thay đổi)
            cache_set(f"company_{symbol}", result, ttl=3600)
            
            return result
            
        except Exception as e:
            return None
    
    # ============================================================
    # MARKET OVERVIEW
    # ============================================================
    def get_market_overview(self) -> Optional[Dict]:
        vn = self.get_stock_price("VNINDEX")
        hnx = self.get_stock_price("HNXINDEX")

        return {
            "vnindex": vn,
            "hnxindex": hnx,
            "time": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        }

    # ============================================================
    # FORMAT
    # ============================================================
    def format_stock_info(self, d: Dict) -> str:
        if not d:
            return "⚠️ Không có dữ liệu"

        change = d["change"]
        color = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"

        return f"""
{color} **{d['symbol']}**  
💰 Giá: {d['price']:,}   ({d['change']:+,} | {d['change_percent']:+.2f}%)  
📊 KL: {d['volume']:,} — High: {d['high']:,} — Low: {d['low']:,}  
🔵 TC: {d['ref_price']:,} — 🔴 Sàn: {d['floor_price']:,} — 🟣 Trần: {d['ceiling_price']:,}  
🕒 {d['time']}
        """.strip()


# ============================================================
# SINGLETON (Giữ nguyên)
# ============================================================
_api_instance = None

def get_vndirect_api() -> VNDirectAPI:
    global _api_instance
    if _api_instance is None:
        _api_instance = VNDirectAPI()
    return _api_instance