## 📊 PHÂN TÍCH MÔ HÌNH THRESHOLD VAR (TVAR)

### 🎯 **Mục đích và Ý nghĩa**

Mô hình TVAR được sử dụng để phân tích **mối quan hệ phi tuyến** giữa cảm xúc tin tức (sentiment score) và lợi suất cổ phiếu (return). Khác với mô hình tuyến tính thông thường, TVAR cho phép:

1. **Phản ứng thị trường khác nhau** tùy theo trạng thái tâm lý (tích cực/tiêu cực)
2. **Regime switching**: Chuyển đổi giữa hai chế độ dựa trên ngưỡng sentiment
3. **Hành vi phi tuyến**: Phản ánh tâm lý bầy đàn và phản ứng bất đối xứng

---

### 🔧 **Cấu trúc Mô hình**

#### **1. Biến số**
```python
# Biến phụ thuộc (dependent variables):
- ret: Log return = log(Pt / Pt-1) 
- score: Sentiment score = (Positive - Negative)

# Biến ngưỡng (threshold variable):
- score(t-1): Sentiment score trễ 1 kỳ
```

#### **2. Chia Regime**
```
Nếu score(t-1) ≤ γ (ngưỡng):  → LOW REGIME (cảm xúc tiêu cực)
Nếu score(t-1) > γ (ngưỡng):   → HIGH REGIME (cảm xúc tích cực)
```

Ngưỡng γ được tính theo **median** hoặc **mean** của sentiment score.

#### **3. Ước lượng VAR cho mỗi Regime**

Mỗi regime có mô hình VAR riêng:

**Low Regime:**
```
ret(t) = α1 + Σ β1,i × ret(t-i) + Σ γ1,i × score(t-i) + ε1(t)
score(t) = α2 + Σ β2,i × ret(t-i) + Σ γ2,i × score(t-i) + ε2(t)
```

**High Regime:** Tương tự với hệ số khác

---

### 📐 **Quy trình Thực hiện**

```python
# Bước 1: Tính threshold (ngưỡng)
threshold = median(sentiment_score)

# Bước 2: Chia dữ liệu theo regime
low_regime = data[score(t-1) ≤ threshold]
high_regime = data[score(t-1) > threshold]

# Bước 3: Chọn lag tối ưu (AIC criterion)
p_low = select_lag_order(low_regime, maxlags=10)
p_high = select_lag_order(high_regime, maxlags=10)

# Bước 4: Ước lượng VAR cho mỗi regime
model_low = VAR(low_regime).fit(p_low)
model_high = VAR(high_regime).fit(p_high)

# Bước 5: Kiểm định chẩn đoán
- Stability check (roots < 1)
- Ljung-Box test (residuals)

# Bước 6: Phân tích IRF (Impulse Response)
irf_low = model_low.irf(steps=15)
irf_high = model_high.irf(steps=15)
```

---

### 📊 **Kết quả và Diễn giải**

#### **Ý nghĩa IRF (Impulse Response Function)**

IRF cho biết **phản ứng của biến** (ret hoặc score) khi có **cú sốc 1 đơn vị** vào biến khác:

1. **Score → Return**: Tin tức tích cực/tiêu cực ảnh hưởng đến giá như thế nào?
2. **Return → Score**: Biến động giá có làm thay đổi sentiment không?

#### **Các Pattern thường gặp**

| Pattern | Ý nghĩa | Hành vi thị trường |
|---------|---------|-------------------|
| **Mean Reversion** | IRF giảm dần về 0 | Giá điều chỉnh về trung bình sau cú sốc |
| **Momentum** | IRF dương trong vài kỳ | Xu hướng tiếp tục ngắn hạn |
| **Overshooting** | IRF tăng mạnh rồi đảo chiều | Phản ứng thái quá, sau đó điều chỉnh |
| **No effect** | IRF gần 0 | Không có tác động đáng kể |

#### **So sánh Low vs High Regime**

Theo kết quả trong code:

- **Low Regime** (tin tiêu cực): Thường có **mean reversion** mạnh hơn → Thị trường điều chỉnh sau tin xấu
- **High Regime** (tin tích cực): Có thể có **momentum ngắn hạn** hoặc **overshooting** → Phản ứng quá mức, sau đó đảo chiều

---


## 🎯 CÁCH TÍNH VÀ XÁC ĐỊNH NGƯỠNG CẢM XÚC CAO/THẤP

### 📊 **1. TÍNH SENTIMENT SCORE**

```python
# Công thức tính điểm cảm xúc
sentiment_score = (Tích_cực) - (Tiêu_cực)
```

**Ý nghĩa:**
- **Score > 0**: Tin tức có xu hướng **tích cực** (nhiều từ tích cực hơn tiêu cực)
- **Score = 0**: Tin tức **trung lập** 
- **Score < 0**: Tin tức có xu hướng **tiêu cực** (nhiều từ tiêu cực hơn tích cực)

**Ví dụ thực tế:**
```
Tin 1: Tích cực = 5, Tiêu cực = 2  → Score = 5 - 2 = +3 (tích cực)
Tin 2: Tích cực = 1, Tiêu cực = 6  → Score = 1 - 6 = -5 (tiêu cực)
Tin 3: Tích cực = 3, Tiêu cực = 3  → Score = 3 - 3 = 0 (trung lập)
```
---
### 🔍 **2. XÁC ĐỊNH NGƯỠNG (THRESHOLD)**

#### **Phương pháp 1: MEDIAN (Mặc định)**
```python
threshold = median(sentiment_score)
```
**Ưu điểm:**
- ✅ **Robust**: Không bị ảnh hưởng bởi outliers (giá trị cực đoan)
- ✅ **Cân bằng**: Chia dữ liệu thành 2 nhóm gần bằng nhau (50%-50%)
- ✅ **Phù hợp thị trường Việt Nam**: Dữ liệu thường có nhiễu, median ổn định hơn

**Ví dụ:**
```
Score: [-8, -5, -2, 0, 1, 3, 5, 7, 10, 15]
Median = (1 + 3) / 2 = 2

→ Low regime: [-8, -5, -2, 0, 1] (5 quan sát ≤ 2)
→ High regime: [3, 5, 7, 10, 15] (5 quan sát > 2)
```
#### **Phương pháp 2: MEAN (Trung bình)**
```python
threshold = mean(sentiment_score)
```

**Ưu điểm:**
- ✅ Phản ánh mức trung bình thực tế
- ⚠️ **Nhạy cảm với outliers**: Một tin rất tích cực/tiêu cực có thể kéo threshold lệch

**Ví dụ:**
```
Score: [-8, -5, -2, 0, 1, 3, 5, 7, 10, 15]
Mean = (-8-5-2+0+1+3+5+7+10+15) / 10 = 2.6

→ Low regime: [-8, -5, -2, 0, 1] (5 quan sát ≤ 2.6)
→ High regime: [3, 5, 7, 10, 15] (5 quan sát > 2.6)
```

---
### 🔄 **3. CHIA REGIME DỰA TRÊN THRESHOLD**

```python
# Sử dụng sentiment TRỄ 1 KỲ (lag_d = 1)
threshold_lagged = sentiment_score.shift(1)

# Chia regime
Low Regime:  score(t-1) ≤ threshold
High Regime: score(t-1) > threshold
```

**Tại sao dùng lag 1?**
- ✅ **Tránh endogeneity**: Sentiment hôm nay không ảnh hưởng đến việc phân loại chính nó
- ✅ **Predictive**: Dùng sentiment hôm qua để dự đoán hành vi giá hôm nay
- ✅ **Realistic**: Thị trường phản ứng sau khi tin tức xuất hiện

---

### 📈 **4. Ý NGHĨA CỦA HAI REGIME**

| Regime | Điều kiện | Ý nghĩa | Hành vi thị trường |
|--------|-----------|---------|-------------------|
| **LOW** | score(t-1) ≤ γ | Cảm xúc **THẤP/TIÊU CỰC** | - Thị trường bi quan<br>- Nhiều tin xấu<br>- Mean reversion mạnh<br>- Điều chỉnh sau panic selling |
| **HIGH** | score(t-1) > γ | Cảm xúc **CAO/TÍCH CỰC** | - Thị trường lạc quan<br>- Nhiều tin tốt<br>- Có thể có momentum<br>- Risk của overshooting |

---

### 🔢 **5. VÍ DỤ CỤ THỂ TỪ PROJECT**

Giả sử dữ liệu FLC có:
```
Date       | Tích cực | Tiêu cực | Score | Score(t-1) | Regime
-----------|----------|----------|-------|------------|--------
2023-01-01 | 3        | 1        | +2    | NaN        | -
2023-01-02 | 2        | 5        | -3    | +2         | HIGH
2023-01-03 | 1        | 8        | -7    | -3         | LOW
2023-01-04 | 4        | 2        | +2    | -7         | LOW
2023-01-05 | 6        | 1        | +5    | +2         | HIGH

Threshold (median) = 0.5
```

**Kết quả:**
- **Low regime** (score ≤ 0.5): 2 quan sát → Giai đoạn tiêu cực, thị trường panic
- **High regime** (score > 0.5): 2 quan sát → Giai đoạn tích cực, thị trường lạc quan

---

### 🎓 **6. CÁC PHƯƠNG PHÁP THRESHOLD NÂNG CAO**

Ngoài median/mean, còn có:

#### **A. Grid Search (Tìm kiếm lưới)**
```python
# Thử nhiều giá trị threshold khác nhau
thresholds = np.percentile(score, [10, 20, 30, 40, 50, 60, 70, 80, 90])
best_threshold = threshold có AIC nhỏ nhất
```

#### **B. Time-Varying Threshold**
```python
# Threshold thay đổi theo thời gian
threshold(t) = rolling_median(score, window=30)
```

#### **C. Conditional Threshold**
```python
# Threshold khác nhau cho bull/bear market
if market == "bull":
    threshold = percentile_75(score)
else:
    threshold = percentile_25(score)
```

#### **D. Multi-Regime (3 chế độ)**
```python
Low:    score ≤ percentile_33
Medium: percentile_33 < score ≤ percentile_67
High:   score > percentile_67
```

---

### 💡 **7. LƯU Ý QUAN TRỌNG**

#### **Sensitivity Analysis**
```python
# Kiểm tra độ nhạy cảm
thresholds = [mean-std, mean, mean+std, median]
for t in thresholds:
    run_tvar_with_threshold(t)
    compare_results()
```

#### **Minimum Observations**
- Mỗi regime cần **ít nhất 40 quan sát** để ước lượng VAR ổn định
- Nếu quá mất cân bằng (90%-10%), nên chọn threshold khác

#### **Interpretation**
- **Threshold ≈ 0**: Điểm giữa tích cực/tiêu cực
- **Threshold > 0**: Thị trường "lạc quan" trung bình
- **Threshold < 0**: Thị trường "bi quan" trung bình

---

### 📊 **8. THỐNG KÊ MẪU TỪ DATA**

Từ kết quả trong notebook:
```
FLC After Scandal:
- Threshold ≈ -0.024 (tiêu cực)
- Low: 212 obs | High: 211 obs (cân bằng)

GAB After Scandal:
- Threshold ≈ 0.031 (tích cực nhẹ)
- Low: 211 obs | High: 210 obs (cân bằng)
```

→ Sau scandal, threshold thường **âm** hoặc **gần 0**, phản ánh tâm lý thị trường tiêu cực

---

### 🚀 **9. RECOMMENDATIONS**

Cho project của bạn:

1. ✅ **Giữ median** làm method mặc định (robust)
2. ✅ Thêm **sensitivity analysis** trong UI
3. ✅ Hiển thị **phân bố score** (histogram) để user hiểu threshold
4. ✅ So sánh **Before vs After Scandal** - threshold thay đổi như thế nào?
5. ✅ Cân nhắc **standardize score** (Z-score) để so sánh giữa các mã

```python
# Z-score standardization
z_score = (score - mean(score)) / std(score)
threshold = 0  # Luôn là 0 sau khi standardize
```


## ⚠️ PHÂN TÍCH: SỬ DỤNG 3 NHÃN PHOBERT CHO TVAR

### 📊 **1. OUTPUT CỦA PHOBERT**

PhoBERT cho ra **3 xác suất** (probabilities), không phải 3 số đếm:

```python
analyze_sentiment("Giá cổ phiếu tăng mạnh")
→ {
    'POSITIVE': 0.85,  # 85% tích cực
    'NEGATIVE': 0.05,  # 5% tiêu cực  
    'NEUTRAL': 0.10    # 10% trung tính
}
# Tổng = 1.0 (100%)
```
- **"tích cực"**: Xác suất tin tức mang tính tích cực (0-1)
- **"tiêu cực"**: Xác suất tin tức mang tính tiêu cực (0-1)
- **"trung tính"**: Xác suất tin tức mang tính trung tính (0-1)
---

### 🔍 **2. CÁCH TVAR ĐANG SỬ DỤNG**

Trong code hiện tại:
```python
# models/tvar_model.py, dòng 160
df["score"] = (
    pd.to_numeric(df.get("tích cực"), errors="coerce")
    - pd.to_numeric(df.get("tiêu cực"), errors="coerce")
)
```

**Formula:** `Score = P(Positive) - P(Negative)`

**Ví dụ:**
```
Tin 1: Pos=0.8, Neg=0.1, Neu=0.1 → Score = 0.8 - 0.1 = +0.7
Tin 2: Pos=0.2, Neg=0.7, Neu=0.1 → Score = 0.2 - 0.7 = -0.5
Tin 3: Pos=0.3, Neg=0.3, Neu=0.4 → Score = 0.3 - 0.3 = 0.0
```

### ✅ **3. ĐÁNH GIÁ: CÁCH HIỆN TẠI CÓ ĐÚNG KHÔNG?**

#### **A. Ưu điểm** ✅

1. **Đơn giản và trực quan**: Score cao = tích cực, thấp = tiêu cực
2. **Phạm vi rõ ràng**: Score ∈ [-1, +1]
   - +1: Hoàn toàn tích cực
   - -1: Hoàn toàn tiêu cực
   - 0: Trung lập hoặc mâu thuẫn
3. **Loại bỏ nhiễu từ Neutral**: Không bị ảnh hưởng bởi tin không quan trọng
4. **Tương thích TVAR**: Threshold dễ xác định (thường gần 0)

#### **B. Hạn chế** ⚠️

1. **BỎ QUA THÔNG TIN NEUTRAL**: 
   - Tin có Neu=0.9 (rất trung lập) được xử lý giống tin Neu=0.1
   - Mất thông tin về độ "không chắc chắn" của model

2. **KHÔNG PHÂN BIỆT ĐỘ TIN CẬY**:
   ```
   Tin A: Pos=0.8, Neg=0.1, Neu=0.1 → Score = +0.7 (rất chắc chắn)
   Tin B: Pos=0.5, Neg=0.2, Neu=0.3 → Score = +0.3 (không chắc chắn)
   ```
   Tin B có uncertainty cao hơn nhưng vẫn được dùng như Tin A

3. **BẤT ĐỐI XỨNG KHI NEU CAO**:
   ```
   Tin 1: Pos=0.4, Neg=0.1, Neu=0.5 → Score = +0.3 (nhưng chủ yếu là neutral!)
   Tin 2: Pos=0.7, Neg=0.4, Neu=0.1 → Score = +0.3 (thực sự mixed sentiment)
   ```
---

### 💡 **4. CÁC PHƯƠNG ÁN TÍNH SCORE TỐT HƠN**

#### **Phương án 1: WEIGHTED SCORE (Khuyên dùng)** 🌟

```python
# Cộng dồn cả 3 nhãn với trọng số
score = (+1 × P(Positive)) + (0 × P(Neutral)) + (-1 × P(Negative))
     = P(Positive) - P(Negative)
```
#### **Phương án 2: NET SENTIMENT với CONFIDENCE**

```python
# Chỉ dùng khi model tự tin (max prob > threshold)
max_prob = max(Pos, Neg, Neu)
if max_prob < 0.5:  # Model không chắc chắn
    score = 0  # Gán neutral
else:
    score = Pos - Neg
```
**Ưu điểm**: Loại bỏ tin tức mơ hồ

#### **Phương án 3: POLARITY STRENGTH**

```python
# Đo "độ cực đoan" của sentiment
polarity = Pos - Neg  # [-1, +1]
strength = 1 - Neu     # [0, 1], càng cao = càng chắc chắn
score = polarity × strength
```
#### **Đề xuất cải tiến:**
**OPTION 1: Giữ nguyên (khuyên dùng cho báo cáo)**
```python
# Đơn giản, dễ giải thích
score = Pos - Neg
```

**OPTION 2: Thêm Polarity Strength (nâng cao)**
```python
# Cân nhắc độ tin cậy của PhoBERT
polarity = Pos - Neg
confidence = 1 - Neu
score = polarity * confidence
```

**OPTION 3: Robustness Check**
```
# So sánh kết quả TVAR với nhiều công thức
methods = {
    'simple': Pos - Neg,
    'weighted': (Pos - Neg) * (1 - Neu),
    'argmax': +1 if argmax==Pos else -1 if argmax==Neg else 0
}

for method, score in methods.items():
    run_tvar(score)
    compare_results()
```
