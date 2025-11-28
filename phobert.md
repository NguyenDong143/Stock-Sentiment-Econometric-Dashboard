# 🤖 PHÂN TÍCH CHI TIẾT CÁCH PHOBERT HOẠT ĐỘNG

### 📚 **1. TỔNG QUAN VỀ PHOBERT**

#### **A. PhoBERT là gì?**

**PhoBERT** (Vietnamese BERT) là mô hình ngôn ngữ dựa trên kiến trúc **BERT (Bidirectional Encoder Representations from Transformers)** được **VinAI Research** phát triển đặc biệt cho **tiếng Việt**.

```
BERT (Google 2018) 
    ↓ Pre-training trên corpus tiếng Việt
PhoBERT-base (VinAI 2020)
    ↓ Fine-tuning cho Sentiment Analysis
wonrax/phobert-base-vietnamese-sentiment (Model bạn đang dùng)
```

---

### 🏗️ **2. KIẾN TRÚC PHOBERT**

#### **A. Cấu trúc Transformer**

```
Input Text
    ↓
[Tokenization] → Tách từ tiếng Việt
    ↓
[Embedding Layer] → Chuyển từ thành vector
    ↓
[12 Transformer Encoder Layers] → Học context bidirectional
    ↓
[Classification Head] → 3 neurons (Pos, Neg, Neu)
    ↓
[Softmax] → Xác suất [0-1]
    ↓
Output: {POSITIVE: 0.7, NEGATIVE: 0.1, NEUTRAL: 0.2}
```

#### **B. Thông số kỹ thuật**

| Thông số | Giá trị | Ý nghĩa |
|----------|---------|---------|
| **Layers** | 12 | Số lớp Transformer encoder |
| **Hidden size** | 768 | Kích thước vector ẩn |
| **Attention heads** | 12 | Số đầu attention trong mỗi lớp |
| **Parameters** | ~135M | Tổng số tham số (triệu) |
| **Vocabulary** | ~64K | Số lượng tokens trong từ điển |
| **Max length** | 256 | Độ dài tối đa (tokens) |

---

### 🔬 **3. QUY TRÌNH XỬ LÝ CHI TIẾT**

#### **BƯỚC 1: TOKENIZATION (Tách từ)**

```python
text = "Cổ phiếu FLC tăng mạnh sau tin tái cấu trúc"

# PhoBERT sử dụng BPE (Byte-Pair Encoding) tokenizer
tokens = tokenizer(text)
```

**Output:**
```python
{
    'input_ids': [0, 5432, 8901, 234, 7654, 3421, ...],  # ID của từng token
    'attention_mask': [1, 1, 1, 1, 1, 1, ...]          # Mask để phân biệt padding
}
```

**Đặc điểm của Vietnamese Tokenizer:**
- Xử lý **tiếng Việt có dấu** chính xác
- Hiểu **từ ghép** (cổ_phiếu, tái_cấu_trúc)
- Xử lý **viết tắt** (VN, HOSE, HNX)
- Nhận diện **số** và **ký hiệu đặc biệt**

---

#### **BƯỚC 2: EMBEDDING (Chuyển thành Vector)**

```python
# 3 loại embedding được cộng lại
Token Embedding:    [768-dim vector] # Ý nghĩa của từ
Position Embedding: [768-dim vector] # Vị trí trong câu
Segment Embedding:  [768-dim vector] # Thuộc câu nào (nếu có nhiều câu)

Final Embedding = Token + Position + Segment
```

**Ví dụ:**
```
"tăng" → [0.23, -0.45, 0.67, ..., 0.12]  (768 số)
"giảm" → [-0.19, 0.52, -0.34, ..., -0.08] (768 số)
```

Các từ có nghĩa gần nhau sẽ có vector gần nhau trong không gian 768 chiều.

---

#### **BƯỚC 3: TRANSFORMER ENCODERS (12 tầng)**

Mỗi tầng Transformer thực hiện:

##### **3a. Multi-Head Self-Attention**
```python
# Mỗi từ "nhìn" toàn bộ câu để hiểu context

"FLC tăng mạnh"
     ↓
Attention weights:
- "FLC" chú ý nhiều đến "tăng" (0.8)
- "tăng" chú ý nhiều đến "mạnh" (0.7)
- "mạnh" chú ý ngược lại "tăng" (0.6)
```

**12 attention heads** học các mối quan hệ khác nhau:
- Head 1: Quan hệ chủ-vị
- Head 2: Quan hệ tính-danh từ
- Head 3: Ngữ nghĩa tích cực/tiêu cực
- ...

##### **3b. Feed-Forward Network**
```python
# Mỗi từ được xử lý qua 2 lớp fully connected
FFN(x) = ReLU(x·W1 + b1)·W2 + b2
```

##### **3c. Layer Normalization + Residual Connection**
```python
# Ổn định quá trình training
output = LayerNorm(x + Attention(x))
output = LayerNorm(output + FFN(output))
```

**Sau 12 tầng**, mỗi từ có vector **context-aware** (hiểu được nghĩa trong câu).

---

#### **BƯỚC 4: CLASSIFICATION HEAD**

```python
# Lấy vector của token [CLS] (đại diện toàn bộ câu)
cls_vector = last_hidden_state[0]  # Shape: (768,)

# Đưa qua fully connected layer
logits = Dense(3)(cls_vector)  # Shape: (3,) - 3 classes

# Output chưa chuẩn hóa
logits = [2.3, -1.5, 0.8]  # Positive cao nhất
```

---

#### **BƯỚC 5: SOFTMAX (Chuyển thành xác suất)**

```python
# Công thức Softmax
probs[i] = exp(logits[i]) / sum(exp(logits[j]) for j in range(3))

# Ví dụ
logits = [2.3, -1.5, 0.8]
         ↓
probs = softmax(logits)
      = [0.71, 0.05, 0.24]  # Tổng = 1.0
```

**Output cuối cùng:**
```python
{
    'POSITIVE': 0.71,  # 71%
    'NEGATIVE': 0.05,  # 5%
    'NEUTRAL': 0.24    # 24%
}
```

---

### 💡 **4. ĐẶC ĐIỂM NỔI BẬT CỦA PHOBERT**

#### **A. Bidirectional (Hai chiều)**

Khác với mô hình truyền thống (đọc từ trái → phải), BERT đọc cả hai chiều:

```
Câu: "Cổ phiếu FLC không tăng mà giảm mạnh"

Unidirectional (LSTM):
→ "không tăng" → Dự đoán POSITIVE (SAI!)

Bidirectional (BERT):
← "giảm mạnh" ← "không tăng" →
→ Hiểu được "không tăng" + "giảm" → NEGATIVE (ĐÚNG!)
```

#### **B. Pre-training + Fine-tuning**

```
Phase 1: PRE-TRAINING (VinAI đã làm sẵn)
- Corpus: 20GB text tiếng Việt (Wiki, báo...)
- Task 1: Masked Language Modeling (MLM)
  "Cổ phiếu [MASK] tăng mạnh" → Dự đoán "FLC"
- Task 2: Next Sentence Prediction (NSP)
  Câu A + Câu B có liên quan không?

Phase 2: FINE-TUNING (wonrax đã làm)
- Dataset: Vietnamese sentiment annotated data
- Task: Phân loại 3 nhãn (Pos/Neg/Neu)
- Training: 10K+ tin tức đã gán nhãn
```

#### **C. Transfer Learning**

PhoBERT đã học được:
- ✅ Ngữ pháp tiếng Việt
- ✅ Quan hệ từ vựng
- ✅ Ngữ cảnh văn hóa Việt Nam

→ Chỉ cần **fine-tune** với ít dữ liệu sentiment là đạt độ chính xác cao!

---

### 🎯 **5. TẠI SAO PHOBERT TỐT CHO TIN TÀI CHÍNH?**

#### **A. Hiểu từ vựng chuyên ngành**

```python
# PhoBERT học được:
"tăng trần" → POSITIVE (technical term)
"về sàn" → NEGATIVE (price floor)
"cổ tức" → POSITIVE (dividend)
"thanh tra" → NEGATIVE (investigation)
"tái cấu trúc" → NEUTRAL/POSITIVE (context-dependent)
```

#### **B. Xử lý phủ định phức tạp**

```python
# Mô hình cũ (rule-based):
"không tốt" → Count("không") → NEGATIVE (ĐÚNG)
"không giảm" → Count("không") → NEGATIVE (SAI!)

# PhoBERT:
"không giảm" → Hiểu "không" + "giảm" → POSITIVE ✅
"không còn tăng" → Context-aware → NEGATIVE ✅
```

#### **C. Độ chính xác cao**

Benchmark trên Vietnamese sentiment:
```
Traditional ML (SVM):     ~75% accuracy
LSTM:                     ~82% accuracy
PhoBERT:                  ~92% accuracy ✅
```

---

### ⚙️ **6. CODE WORKFLOW TRONG PROJECT**

```python
# 1. INITIALIZATION (Chỉ chạy 1 lần)
tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
model = AutoModelForSequenceClassification.from_pretrained(...)
model.eval()  # Chế độ inference (không training)

# 2. INFERENCE (Mỗi lần phân tích)
def analyze_sentiment(text):
    # a. Tokenization
    inputs = tokenizer(text, 
                      return_tensors="pt",    # PyTorch tensor
                      truncation=True,        # Cắt nếu quá dài
                      padding=True)           # Padding đến max_length
    
    # b. Forward pass (không tính gradient)
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.logits: Tensor([2.3, -1.5, 0.8])
    
    # c. Softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # probs: Tensor([0.71, 0.05, 0.24])
    
    # d. Map to labels
    label_map = model.config.id2label  # {0: 'NEG', 1: 'NEU', 2: 'POS'}
    result = {label_map[i]: probs[0][i].item() for i in range(3)}
    
    return result
```
