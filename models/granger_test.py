# ============================================================
# 📘 models/granger_test.py — Phiên bản cải tiến
# ============================================================
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR


@st.cache_data(show_spinner="Đang chạy kiểm định Granger...")
def granger_test(
    df: pd.DataFrame, 
    columns_to_test: list, 
    maxlags: int = 14,
    significance_level: float = 0.05,
    test_individually: bool = False
):
    """
    Thực hiện kiểm định nhân quả Granger đa biến (VAR-based) theo phương pháp trong paper.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dữ liệu chứa các biến cần kiểm định
    columns_to_test : list
        Danh sách các biến cần phân tích
    maxlags : int
        Số lag tối đa để chọn (mặc định: 14)
    significance_level : float
        Mức ý nghĩa thống kê (mặc định: 0.05)
    test_individually : bool
        Nếu True, test từng biến riêng lẻ; False = test tất cả cùng lúc
    
    Returns:
    --------
    results_df : pd.DataFrame
        Bảng kết quả kiểm định
    var_model : VAR
        Mô hình VAR đã ước lượng (để sử dụng cho TVAR sau này)
    """

    print("\n" + "="*80)
    print("🔁 KIỂM ĐỊNH NHÂN QUẢ GRANGER (VAR-BASED)")
    print("="*80)

    if df.empty:
        print("⚠️ Dữ liệu rỗng — không thể kiểm định.")
        return pd.DataFrame(), None

    df = df.copy()
    results = []
    stationary_vars = []
    transformation_info = {}

    # =======================
    # BƯỚC 1: KIỂM TRA TÍNH DỪNG (Stationarity Check)
    # =======================
    print("\n📌 BƯỚC 1: Kiểm tra tính dừng (Augmented Dickey-Fuller Test)")
    print("-" * 80)
    
    for column in columns_to_test:
        if column not in df.columns:
            print(f"⚠️ Cột '{column}' không tồn tại trong dữ liệu!")
            continue

        series = df[column].dropna()
        if len(series) < 10:
            print(f"⚠️ Dữ liệu '{column}' quá ít để kiểm định ADF (cần ít nhất 10 quan sát).")
            continue

        try:
            # Thực hiện ADF test
            adf_result = adfuller(series, autolag='AIC')
            adf_stat = adf_result[0]
            p_value = adf_result[1]
            critical_values = adf_result[4]
            
            if p_value > significance_level:
                # Chuỗi KHÔNG dừng → lấy sai phân
                diff_col = f"{column}_diff"
                df[diff_col] = df[column].diff()
                stationary_vars.append(diff_col)
                transformation_info[column] = {
                    'original': column,
                    'transformed': diff_col,
                    'method': 'first_difference',
                    'adf_statistic': adf_stat,
                    'p_value': p_value
                }
                print(f"❌ {column:20s} | ADF={adf_stat:8.3f} | p={p_value:.4f} | KHÔNG DỪNG")
                print(f"   → Áp dụng sai phân bậc 1: {diff_col}")
            else:
                # Chuỗi dừng
                stationary_vars.append(column)
                transformation_info[column] = {
                    'original': column,
                    'transformed': column,
                    'method': 'none',
                    'adf_statistic': adf_stat,
                    'p_value': p_value
                }
                print(f"✅ {column:20s} | ADF={adf_stat:8.3f} | p={p_value:.4f} | DỪNG")
                
        except Exception as e:
            print(f"❌ Lỗi kiểm định ADF cho '{column}': {e}")

    # Loại bỏ dòng có NaN sau khi sai phân
    df_var = df[stationary_vars].dropna()
    
    if df_var.empty:
        print("\n❌ THẤT BẠI: Không còn dữ liệu hợp lệ sau khi xử lý sai phân.")
        return pd.DataFrame(), None

    print(f"\n✅ Tóm tắt:")
    print(f"   - Số biến dừng: {len(stationary_vars)}")
    print(f"   - Số quan sát hợp lệ: {len(df_var)}")
    print(f"   - Các biến trong mô hình: {', '.join(stationary_vars)}")

    # =======================
    # BƯỚC 2: CHỌN ĐỘ TRỄ TỐI ƯU (Lag Selection)
    # =======================
    print("\n📌 BƯỚC 2: Chọn độ trễ tối ưu")
    print("-" * 80)
    
    try:
        model = VAR(df_var)
        lag_selection = model.select_order(maxlags=maxlags)
        
        # Lấy lag theo AIC
        best_lag = lag_selection.selected_orders.get("aic", 5)
        
        if not isinstance(best_lag, int) or best_lag < 1:
            best_lag = 5
            print(f"⚠️ Lag không hợp lệ, sử dụng mặc định: {best_lag}")
        
        # Hiển thị các tiêu chí
        print(f"✅ Lag được chọn (theo AIC): {best_lag}")
        print(f"\n   Bảng tiêu chí thông tin:")
        print(f"   {'Lag':>5} | {'AIC':>12} | {'BIC':>12} | {'FPE':>12} | {'HQIC':>12}")
        print(f"   {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
        
        # Sửa lỗi: kiểm tra xem lag_selection.aic là dict hay không
        try:
            if isinstance(lag_selection.aic, dict):
                max_display_lag = min(maxlags, max(lag_selection.aic.keys()))
            else:
                max_display_lag = min(maxlags, 10)  # fallback
            
            for lag in range(1, max_display_lag + 1):
                aic_val = lag_selection.aic.get(lag, np.nan) if isinstance(lag_selection.aic, dict) else np.nan
                bic_val = lag_selection.bic.get(lag, np.nan) if isinstance(lag_selection.bic, dict) else np.nan
                fpe_val = lag_selection.fpe.get(lag, np.nan) if isinstance(lag_selection.fpe, dict) else np.nan
                hqic_val = lag_selection.hqic.get(lag, np.nan) if isinstance(lag_selection.hqic, dict) else np.nan
                
                marker = " ←" if lag == best_lag else ""
                print(f"   {lag:>5} | {aic_val:>12.2f} | {bic_val:>12.2f} | {fpe_val:>12.6f} | {hqic_val:>12.2f}{marker}")
        except Exception as e:
            print(f"   ⚠️ Không thể hiển thị chi tiết tiêu chí: {e}")
        
    except Exception as e:
        print(f"⚠️ Lỗi khi chọn lag tối ưu: {e}")
        best_lag = 5
        print(f"   Sử dụng lag mặc định: {best_lag}")

    # Kiểm tra số quan sát đủ cho mô hình
    min_obs_required = best_lag + 10
    if len(df_var) < min_obs_required:
        print(f"\n❌ THẤT BẠI: Không đủ dữ liệu!")
        print(f"   Cần ít nhất: {min_obs_required} quan sát")
        print(f"   Hiện có:     {len(df_var)} quan sát")
        return pd.DataFrame(), None

    # =======================
    # BƯỚC 3: ƯỚC LƯỢNG MÔ HÌNH VAR
    # =======================
    print("\n📌 BƯỚC 3: Ước lượng mô hình VAR")
    print("-" * 80)
    
    try:
        var_model = model.fit(best_lag)
        print(f"✅ Mô hình VAR({best_lag}) đã được ước lượng thành công.")
        
        # Kiểm tra tính ổn định của mô hình
        if hasattr(var_model, 'is_stable'):
            is_stable = var_model.is_stable()
            if is_stable:
                print("✅ Mô hình VAR ổn định (tất cả eigenvalues < 1)")
            else:
                print("⚠️ CẢNH BÁO: Mô hình VAR KHÔNG ỔN ĐỊNH!")
                print("   Kết quả có thể không đáng tin cậy.")
        
        # Thông tin mô hình
        print(f"\n   Thông tin mô hình:")
        print(f"   - Số phương trình: {var_model.neqs}")
        print(f"   - Số quan sát:     {var_model.nobs}")
        print(f"   - Số tham số:      {var_model.nobs * var_model.neqs}")
        
    except Exception as e:
        print(f"❌ THẤT BẠI: Lỗi khi ước lượng mô hình VAR: {e}")
        return pd.DataFrame(), None

    # =======================
    # BƯỚC 4: KIỂM ĐỊNH NHÂN QUẢ GRANGER
    # =======================
    print("\n📌 BƯỚC 4: Kiểm định nhân quả Granger")
    print("-" * 80)
    
    if test_individually:
        # Test từng cặp biến riêng lẻ (theo paper Tables IV, V)
        print("🔍 Chế độ: Test từng biến riêng lẻ (pairwise)")
        print()
        
        for caused in df_var.columns:
            print(f"\n📊 Biến bị ảnh hưởng: {caused}")
            print(f"   {'-'*70}")
            
            for causing_var in [c for c in df_var.columns if c != caused]:
                try:
                    test = var_model.test_causality(
                        caused=caused, 
                        causing=[causing_var], 
                        kind='f'
                    )
                    
                    f_stat = round(test.test_statistic, 4)
                    p_value = round(test.pvalue, 4)
                    is_significant = p_value < significance_level
                    conclusion = "✅ Có nhân quả" if is_significant else "❌ Không có nhân quả"
                    
                    # Tính hệ số trung bình CHÍNH XÁC
                    mean_coef = _calculate_mean_coefficient(
                        var_model, caused, [causing_var], best_lag
                    )
                    
                    results.append({
                        "Biến bị ảnh hưởng": caused,
                        "Biến gây ảnh hưởng": causing_var,
                        "Lag": best_lag,
                        "Coef (TB)": mean_coef,
                        "F-statistic": f_stat,
                        "p-value": p_value,
                        "Có ý nghĩa": "✅" if is_significant else "❌",
                        "Kết luận": conclusion
                    })
                    
                    # Hiển thị kết quả
                    sig_marker = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                    print(f"   {causing_var:20s} → F={f_stat:8.2f} | p={p_value:.4f}{sig_marker:3s} | Coef={mean_coef:8.4f} | {conclusion}")
                    
                except Exception as e:
                    print(f"   ⚠️ Lỗi test {causing_var}: {e}")
    
    else:
        # Test tất cả biến khác cùng lúc (mặc định)
        print("🔍 Chế độ: Test tất cả biến cùng lúc (joint test)")
        print()
        
        for caused in df_var.columns:
            causing = [c for c in df_var.columns if c != caused]
            
            try:
                test = var_model.test_causality(
                    caused=caused, 
                    causing=causing, 
                    kind='f'
                )
                
                f_stat = round(test.test_statistic, 4)
                p_value = round(test.pvalue, 4)
                is_significant = p_value < significance_level
                conclusion = "✅ Có quan hệ nhân quả" if is_significant else "❌ Không có quan hệ"
                
                # Tính hệ số trung bình CHÍNH XÁC
                mean_coef = _calculate_mean_coefficient(
                    var_model, caused, causing, best_lag
                )
                
                results.append({
                    "Biến bị ảnh hưởng": caused,
                    "Biến gây ảnh hưởng": ", ".join(causing),
                    "Lag": best_lag,
                    "Coef (TB)": mean_coef,
                    "F-statistic": f_stat,
                    "p-value": p_value,
                    "Có ý nghĩa": "✅" if is_significant else "❌",
                    "Kết luận": conclusion
                })
                
                print(f"\n📊 {caused} ← [{', '.join(causing)}]:")
                print(f"   F-statistic = {f_stat:.4f}")
                print(f"   p-value     = {p_value:.4f}")
                print(f"   Coefficient = {mean_coef:.6f}")
                print(f"   → {conclusion}")
                
            except Exception as e:
                print(f"\n⚠️ Lỗi kiểm định nhân quả cho '{caused}': {e}")

    # =======================
    # BƯỚC 5: TÓM TẮT KẾT QUẢ
    # =======================
    print("\n" + "="*80)
    print("📋 TÓM TẮT KẾT QUẢ")
    print("="*80)
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Đếm số quan hệ có ý nghĩa
        significant_count = len(results_df[results_df['p-value'] < significance_level])
        total_count = len(results_df)
        
        print(f"✅ Hoàn thành: {total_count} kiểm định")
        print(f"✅ Có ý nghĩa thống kê (p < {significance_level}): {significant_count}/{total_count}")
        
        if significant_count > 0:
            print(f"\n📊 Các quan hệ nhân quả có ý nghĩa:")
            sig_results = results_df[results_df['p-value'] < significance_level]
            for _, row in sig_results.iterrows():
                print(f"   • {row['Biến gây ảnh hưởng']} → {row['Biến bị ảnh hưởng']} (p={row['p-value']:.4f})")
        
        print("="*80)
        return results_df, var_model
    else:
        print("❌ Không có kết quả hợp lệ.")
        print("="*80)
        return pd.DataFrame(), None


def _calculate_mean_coefficient(var_model, caused: str, causing: list, best_lag: int):
    """
    Tính hệ số trung bình CHÍNH XÁC của các biến causing trong phương trình caused.
    
    Parameters:
    -----------
    var_model : VAR
        Mô hình VAR đã ước lượng
    caused : str
        Tên biến bị ảnh hưởng (biến phụ thuộc)
    causing : list
        Danh sách biến gây ảnh hưởng
    best_lag : int
        Số lag được sử dụng trong mô hình
    
    Returns:
    --------
    float : Hệ số trung bình tuyệt đối
    """
    try:
        # Lấy tất cả hệ số của phương trình caused
        equation_params = var_model.params[caused]
        
        causing_coefs = []
        
        # Với mỗi biến causing, lấy tất cả lag của nó
        for var in causing:
            for lag in range(1, best_lag + 1):
                lag_name = f"L{lag}.{var}"
                if lag_name in equation_params.index:
                    coef_value = equation_params[lag_name]
                    causing_coefs.append(abs(coef_value))  # Lấy giá trị tuyệt đối
        
        if causing_coefs:
            return round(np.mean(causing_coefs), 6)
        else:
            return 0.0
            
    except Exception as e:
        print(f"⚠️ Lỗi tính coefficient: {e}")
        return 0.0


def perform_granger_analysis(
    sentiment_scores: pd.Series, 
    stock_prices: pd.Series, 
    maxlags: int = 14,
    test_individually: bool = False,
    significance_level: float = 0.05
):
    """
    Wrapper function để thực hiện phân tích Granger hoàn chỉnh.
    
    Parameters:
    -----------
    sentiment_scores : pd.Series
        Chuỗi điểm sentiment (index là ngày)
    stock_prices : pd.Series
        Chuỗi giá đóng cửa (index là ngày)
    maxlags : int
        Số lag tối đa
    test_individually : bool
        Test từng biến riêng hay không
    significance_level : float
        Mức ý nghĩa thống kê
    
    Returns:
    --------
    results_df : pd.DataFrame
        Bảng kết quả
    var_model : VAR
        Mô hình VAR
    """
    # Tạo DataFrame kết hợp
    df = pd.DataFrame({
        'sentiment_score': sentiment_scores,
        'stock_price': stock_prices
    })
    
    # Loại bỏ NaN
    df = df.dropna()
    
    if len(df) < 30:
        print("⚠️ Không đủ dữ liệu để phân tích (cần ít nhất 30 quan sát)")
        return pd.DataFrame(), None
    
    # Thực hiện kiểm định
    results_df, var_model = granger_test(
        df=df,
        columns_to_test=['sentiment_score', 'stock_price'],
        maxlags=maxlags,
        test_individually=test_individually,
        significance_level=significance_level
    )
    
    return results_df, var_model