# ============================================================
# 📘 models/tvar_model.py — Threshold Vector Autoregression
# ============================================================
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings("ignore")


class ThresholdVAR:
    """
    Threshold Vector Autoregression Model
    Theo methodology trong paper (Section III.G.5.c)
    """

    def __init__(self, data: pd.DataFrame, threshold_var: str, dependent_vars: list):
        self.data = data.copy()
        self.threshold_var = threshold_var
        self.dependent_vars = dependent_vars
        self.threshold_value = None
        self.regime_low = None
        self.regime_high = None
        self.model_low = None
        self.model_high = None
        self.results = {}

    # ============================================================
    # 🔹 1. Xác định ngưỡng (threshold value)
    # ============================================================
    def calculate_threshold(self, method="median"):
        if method == "median":
            self.threshold_value = self.data[self.threshold_var].median()
        elif method == "mean":
            self.threshold_value = self.data[self.threshold_var].mean()
        else:
            raise ValueError("Method phải là 'median' hoặc 'mean'")
        print(f"✅ Threshold value (γ) = {self.threshold_value:.6f} ({method})")
        return self.threshold_value

    # ============================================================
    # 🔹 2. Chia dữ liệu thành hai Regime
    # ============================================================
    def split_regimes(self, lag_d: int = 1):
        if self.threshold_value is None:
            self.calculate_threshold()

        threshold_lagged = self.data[self.threshold_var].shift(lag_d)
        mask_low = threshold_lagged <= self.threshold_value
        mask_high = threshold_lagged > self.threshold_value

        self.regime_low = self.data.loc[mask_low].dropna()
        self.regime_high = self.data.loc[mask_high].dropna()

        print(f"\n📊 Phân chia Regime:")
        print(f"   Regime 1 (Low):  {len(self.regime_low)} quan sát")
        print(f"   Regime 2 (High): {len(self.regime_high)} quan sát")

        return self.regime_low, self.regime_high

    # ============================================================
    # 🔹 3. Chọn số bậc trễ tối ưu (lag order)
    # ============================================================
    def select_lag_order(self, regime_data: pd.DataFrame, maxlags: int = 10):
        try:
            model = VAR(regime_data[self.dependent_vars])
            lag_order = model.select_order(maxlags=maxlags)
            selected_lag = lag_order.selected_orders.get("aic") or 1
            print(f"📈 Lag tối ưu (AIC) = {selected_lag}")
            return selected_lag
        except Exception as e:
            print("⚠️ Lỗi khi chọn lag:", e)
            return 1

    # ============================================================
    # 🔹 4. Ước lượng mô hình VAR cho từng Regime
    # ============================================================
    def fit(self, maxlags=10):
        if self.regime_low is None or self.regime_high is None:
            self.split_regimes()

        # LOW regime
        try:
            p_low = self.select_lag_order(self.regime_low, maxlags=maxlags)
            model_low = VAR(self.regime_low[self.dependent_vars])
            self.model_low = model_low.fit(p_low)
            print(f"✅ LOW regime fitted (lag={p_low})")
        except Exception as e:
            print("❌ Lỗi khi ước lượng LOW regime:", e)
            self.model_low = None

        # HIGH regime
        try:
            p_high = self.select_lag_order(self.regime_high, maxlags=maxlags)
            model_high = VAR(self.regime_high[self.dependent_vars])
            self.model_high = model_high.fit(p_high)
            print(f"✅ HIGH regime fitted (lag={p_high})")
        except Exception as e:
            print("❌ Lỗi khi ước lượng HIGH regime:", e)
            self.model_high = None

        return self.model_low, self.model_high

    # ============================================================
    # 🔹 5. Diagnostic kiểm tra ổn định
    # ============================================================
    def diagnostics(self):
        def check_var(model, name):
            if model is None:
                return f"{name}: Model not estimated", []
            stable = model.is_stable()
            max_root = np.max(np.abs(model.roots))
            ljung = []
            for col in model.resid.columns:
                lb = acorr_ljungbox(model.resid[col], lags=[10], return_df=True)
                ljung.append((col, float(lb["lb_pvalue"].iloc[0])))
            msg = f"{name}: stable={stable}, max_root={max_root:.3f}"
            return msg, ljung

        diag_low, lb_low = check_var(self.model_low, "LOW")
        diag_high, lb_high = check_var(self.model_high, "HIGH")
        self.results["diagnostics"] = {"low": (diag_low, lb_low), "high": (diag_high, lb_high)}
        return self.results["diagnostics"]

    # ============================================================
    # 🔹 6. Impulse Response Function
    # ============================================================
    def impulse_response(self, steps=15):
        irf_low = self.model_low.irf(steps) if self.model_low else None
        irf_high = self.model_high.irf(steps) if self.model_high else None
        self.results["irf_low"] = irf_low
        self.results["irf_high"] = irf_high
        return irf_low, irf_high

    # ============================================================
    # 🔹 7. Xuất summary (tóm tắt kết quả)
    # ============================================================
    def summary(self):
        print("\n========== SUMMARY ==========")
        if self.model_low:
            print("\n--- LOW REGIME ---")
            print(self.model_low.summary())
        if self.model_high:
            print("\n--- HIGH REGIME ---")
            print(self.model_high.summary())
        self.diagnostics()
        return self.results


# ============================================================
# 🧠 HÀM CHẠY TVAR DÙNG CHO DASHBOARD STREAMLIT
# ============================================================
@st.cache_data(show_spinner="Đang chạy mô hình TVAR...")
def run_tvar(df: pd.DataFrame, ticker: str, steps: int = 15):
    df = df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ret"] = np.log(df["close"].replace(0, np.nan)).diff()
    df["score"] = (
        pd.to_numeric(df.get("tích cực"), errors="coerce")
        - pd.to_numeric(df.get("tiêu cực"), errors="coerce")
    )
    df = df[["ret", "score"]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < 40:
        return {"error": f"Dữ liệu quá nhỏ ({len(df)} quan sát) cho {ticker}"}

    tvar = ThresholdVAR(df, threshold_var="score", dependent_vars=["ret", "score"])
    tvar.calculate_threshold()
    tvar.split_regimes(lag_d=1)
    tvar.fit(maxlags=6)
    diagnostics = tvar.diagnostics()
    irf_low, irf_high = tvar.impulse_response(steps=steps)

    # ✅ Sửa lỗi: dùng str() thay vì .as_text()
    results = {
        "ticker": ticker,
        "threshold": float(tvar.threshold_value),
        "low_n": len(tvar.regime_low),
        "high_n": len(tvar.regime_high),
        "low": {
            "lag": tvar.model_low.k_ar if tvar.model_low else None,
            "summary": str(tvar.model_low.summary()) if tvar.model_low else "N/A",
            "diag": diagnostics["low"][0],
            "irf": irf_low,
        },
        "high": {
            "lag": tvar.model_high.k_ar if tvar.model_high else None,
            "summary": str(tvar.model_high.summary()) if tvar.model_high else "N/A",
            "diag": diagnostics["high"][0],
            "irf": irf_high,
        },
    }

    return results
