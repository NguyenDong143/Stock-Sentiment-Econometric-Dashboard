# ======================================================
# 📦 PHOBERT SENTIMENT ANALYSIS (WONRAX VERSION)
# ======================================================

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

# ------------------------------------------------------
# 1️⃣ Load model và tokenizer (Wonrax fine-tuned PhoBERT)
# ------------------------------------------------------
MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"

@st.cache_resource(show_spinner=False)
def load_phobert_model():
    """Cache PhoBERT model để tránh load lại mỗi lần chạy"""
    # Sử dụng use_fast=True cho tokenizer nhanh hơn
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        use_fast=True,  # Tokenizer nhanh hơn
        model_max_length=256  # Giới hạn độ dài input
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # Set eval mode
    
    # Tắt gradient để tối ưu memory
    for param in model.parameters():
        param.requires_grad = False
    
    return tokenizer, model

def get_model():
    """Lazy getter cho model"""
    return load_phobert_model()

# ------------------------------------------------------
# 2️⃣ Phân tích cảm xúc cho 1 văn bản
# ------------------------------------------------------
def analyze_sentiment(text: str):
    if not isinstance(text, str):
        text = str(text)

    # Lazy load model
    tokenizer, model = get_model()
    
    # Tokenize với max_length giới hạn
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256,  # Giới hạn độ dài input
        padding="max_length"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Map nhãn sang cảm xúc
    label_map = model.config.id2label
    result = {label_map[i]: probs[0][i].item() for i in range(len(probs[0]))}
    return result

# ------------------------------------------------------
# 3️⃣ Hàm xử lý DataFrame
# ------------------------------------------------------
def analyze_dataframe(df: pd.DataFrame, column: str):
    if column not in df.columns:
        raise ValueError(f"❌ Cột '{column}' không tồn tại trong DataFrame!")

    df[column] = df[column].fillna("").astype(str)
    results = [analyze_sentiment(text) for text in df[column]]
    return pd.DataFrame(results)

# ------------------------------------------------------
# 4️⃣ Hàm phân loại nhanh cho Streamlit
# ------------------------------------------------------
def classify_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]

    results = []
    for text in texts:
        output = analyze_sentiment(text)
        label = max(output, key=output.get).lower()

        if "neg" in label:
            results.append(-1)
        elif "neu" in label:
            results.append(0)
        else:
            results.append(1)
    return results

# ------------------------------------------------------
# 5️⃣ Test nhanh
# ------------------------------------------------------
if __name__ == "__main__":
    text = "Thị trường chứng khoán giảm mạnh, khối ngoại bán ròng hàng trăm tỷ đồng."
    print(analyze_sentiment(text))
