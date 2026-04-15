# 📊 Analyzing Effects of News on Stock Prices using NLP and Econometric Models

### *Evidence from the Vietnamese Stock Market*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PhoBERT-NLP-00D4AA?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FinBERT-Finance-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge"/>
</p>

---

## 🎯 Abstract

This project investigates the relationship between **financial news sentiment** and **stock price dynamics** in the Vietnamese stock market.

Using **PhoBERT (Vietnamese NLP)** for sentiment extraction and **econometric models** (Pearson correlation, Granger causality, and Threshold VAR), the study evaluates whether sentiment signals can explain or predict stock price movements, particularly in potentially manipulated stocks.

---

## 🧠 Research Objective

* Analyze how **news sentiment influences stock price behavior**
* Detect **potential manipulation patterns**
* Evaluate **predictive power of sentiment signals**

---

## 🔬 Methodology

### 1. Data Collection

* 135,000+ financial news articles (VnEconomy)
* Historical stock prices (2018–2023)

### 2. NLP Processing

* PhoBERT → sentiment classification (positive, neutral, negative)
* Sentiment encoding: -1, 0, +1
* Daily aggregation of sentiment signals

### 3. Data Integration

* Align sentiment scores with stock prices
* Construct time-series dataset

### 4. Econometric Models

* **Pearson Correlation** → linear relationship
* **Granger Causality** → predictive relationship
* **Threshold VAR (TVAR)** → nonlinear regime dynamics

---

## 📊 System Pipeline

![Pipeline](./images/pipeline.png)

```text
News Data + Stock Data
        ↓
Data Collection & Cleaning
        ↓
PhoBERT Sentiment Analysis
        ↓
Sentiment Aggregation
        ↓
Time-Series Integration
        ↓
Econometric Analysis
        ↓
Visualization Dashboard
```

---

## 📈 Key Findings

* Sentiment exhibits **statistically significant correlation** with stock prices
* Evidence of **Granger causality** in multiple stocks
* Strong **nonlinear dynamics** captured by TVAR
* Sentiment impact weakens after major market events (post-2022)

---

## 🖥️ System Implementation

Although research-oriented, the project includes a full implementation:

### 📊 Dashboard

![Dashboard](./images/dashboard.png)

### 🤖 Prediction Module

![Prediction](./images/prediction.png)

---

## 🛠 Technologies

* **NLP:** PhoBERT, FinBERT
* **Econometrics:** Statsmodels (VAR, TVAR, Granger)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Streamlit, Plotly

---

## 📄 Research Paper

Full paper available:
👉 `./paper/research_paper.pdf`

---

## ▶️ Run the Project

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Work

* Integrate **real-time sentiment data**
* Apply **deep learning models (LSTM, Transformers)**
* Extend to **other emerging markets**

---

## 👤 Author

Nguyen Hoang Dong

---

<p align="center">
⭐ If you find this research useful, please give it a star!
</p>
