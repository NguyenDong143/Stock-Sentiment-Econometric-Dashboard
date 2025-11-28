import requests
import json

# Thử nhiều endpoint khác nhau
endpoints = [
    "https://finfo-api.vndirect.com.vn/stocks",
    "https://finfo-api.vndirect.com.vn/v3/stocks",
    "https://finfo-api.vndirect.com.vn/v4/stocks",
    "https://finfo-api.vndirect.com.vn/stocks/company",
    "https://finfo-api.vndirect.com.vn/v3/stocks/company",
]

symbol = "VCB"

for endpoint in endpoints:
    print(f"\n{'='*60}")
    print(f"Testing: {endpoint}")
    print(f"{'='*60}")
    
    try:
        r = requests.get(
            endpoint,
            params={"q": f"code:{symbol}", "size": 1},
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        print(f"✅ Status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"Keys: {list(data.keys())}")
            if data.get('data'):
                print(f"Sample data keys: {list(data['data'][0].keys())[:10]}")
        else:
            print(f"Response: {r.text[:200]}")
            
    except requests.Timeout:
        print("❌ Timeout")
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
