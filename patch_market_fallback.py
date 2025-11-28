"""
Patch file - Add fallback for market overview when API returns None
Apply this by copying the method into chatbot_services.py
"""

def generate_response_stream_with_fallback(self, user_message: str, context: Optional[str] = None):
    """Generate streaming response với fallback khi API không có data"""
    try:
        detected_symbols = self._extract_stock_symbols(user_message)
        context_blocks = []
        
        # KIỂM TRA CÂU HỎI TỔNG QUAN THỊ TRƯỜNG
        if self._is_market_overview_query(user_message) and not detected_symbols:
            market_data_available = False
            
            # Thử lấy từ API
            try:
                api = get_vndirect_api()
                overview = api.get_market_overview()
                
                if overview and (overview.get('vnindex') or overview.get('hnxindex')):
                    vn_data = overview.get('vnindex')
                    hnx_data = overview.get('hnxindex')
                    
                    market_info = "📊 TỔNG QUAN THỊ TRƯỜNG HÔM NAY:\n"
                    
                    if vn_data:
                        change_icon = "🔺" if vn_data['change'] > 0 else "🔻" if vn_data['change'] < 0 else "➡️"
                        market_info += f"{change_icon} **VNINDEX**: {vn_data['price']:,.2f} ({vn_data['change']:+,.2f} | {vn_data['change_percent']:+.2f}%)\n"
                        market_data_available = True
                    
                    if hnx_data:
                        change_icon = "🔺" if hnx_data['change'] > 0 else "🔻" if hnx_data['change'] < 0 else "➡️"
                        market_info += f"{change_icon} **HNXINDEX**: {hnx_data['price']:,.2f} ({hnx_data['change']:+,.2f} | {hnx_data['change_percent']:+.2f}%)\n"
                        market_data_available = True
                    
                    if market_data_available:
                        market_info += f"\n⏰ Cập nhật: {overview.get('time', '')}"
                        context_blocks.append(market_info)
            
            except Exception as e:
                logger.warning(f"API error: {e}")
            
            # FALLBACK: Dùng historical data
            if not market_data_available:
                fallback = "📊 PHÂN TÍCH THỊ TRƯỜNG:\n\n"
                fallback += "⚠️ *Dữ liệu realtime tạm thời không khả dụng. Phân tích dựa trên xu hướng gần đây.*\n\n"
                
                try:
                    top_stocks = ["VCB", "BID", "HPG", "VHM", "FPT"]
                    stock_data = []
                    
                    for symbol in top_stocks:
                        df = load_price_data(symbol)
                        if not df.empty and len(df) >= 5:
                            current_price = df['close'].iloc[-1]
                            prev_price = df['close'].iloc[-2]
                            change_pct = ((current_price - prev_price) / prev_price) * 100
                            
                            week_start = df['close'].iloc[-5]
                            week_change = ((current_price - week_start) / week_start) * 100
                            
                            icon = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "🟡"
                            stock_data.append({
                                'symbol': symbol,
                                'change': change_pct,
                                'week_change': week_change,
                                'icon': icon,
                                'price': current_price
                            })
                    
                    if stock_data:
                        fallback += "📈 **NHÓM BLUE CHIPS (Phiên gần nhất):**\n"
                        for stock in stock_data:
                            fallback += f"{stock['icon']} **{stock['symbol']}**: {stock['price']:,.0f} VNĐ "
                            fallback += f"({stock['change']:+.2f}% phiên | {stock['week_change']:+.2f}% tuần)\n"
                        
                        # Phân tích xu hướng chung
                        avg_change = sum(s['change'] for s in stock_data) / len(stock_data)
                        avg_week = sum(s['week_change'] for s in stock_data) / len(stock_data)
                        
                        fallback += f"\n📊 **Trung bình nhóm**: {avg_change:+.2f}% (phiên) | {avg_week:+.2f}% (tuần)\n"
                        
                        if avg_change > 1.0:
                            fallback += "\n💡 **Xu hướng**: Nhóm blue chips tích cực, thị trường có động lực tăng."
                        elif avg_change < -1.0:
                            fallback += "\n💡 **Xu hướng**: Nhóm blue chips điều chỉnh, thị trường có áp lực bán."
                        else:
                            fallback += "\n💡 **Xu hướng**: Nhóm blue chips dao động nhẹ, thị trường đi ngang."
                        
                        context_blocks.append(fallback)
                    else:
                        fallback += "⚠️ Không thể phân tích chi tiết. Vui lòng thử lại sau hoặc hỏi về mã cụ thể (VD: 'VCB hôm nay?')"
                        context_blocks.append(fallback)
                
                except Exception as e2:
                    logger.error(f"Fallback analysis failed: {e2}")
                    fallback += "\n⚠️ Không thể phân tích. Hãy hỏi về mã cụ thể: 'Phân tích VCB?'"
                    context_blocks.append(fallback)
        
        # Continue with rest of the method...
        # (existing code for symbol-specific queries)
        
        realtime_prices_markdown = "\n\n".join(context_blocks) if context_blocks else None
        
        # ... rest of method
