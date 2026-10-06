import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình giao diện Streamlit rộng rãi, đẹp mắt hơn
st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="📈",
    layout="wide"
)

# 1. Hàm load model và encoders an toàn với cache
@st.cache_resource
def load_artifacts():
    try:
        with open("xgboost_demand_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        # Load thêm dataset gốc nếu có để phục vụ phần vẽ chart phân tích
        df_analysis = pd.read_csv("demand_forecasting.csv")
        return model, encoders, df_analysis
    except FileNotFoundError:
        return None, None, None

model, label_encoders, df = load_artifacts()

# Kiểm tra nếu chưa train/lưu file artifact
if model is None or label_encoders is None:
    st.error("⚠️ Không tìm thấy file mô hình (`xgboost_demand_model.pkl`) hoặc `label_encoders.pkl`. Vui lòng chạy code huấn luyện mô hình trước!")
    st.stop()

# Tiêu đề ứng dụng
st.title("📦 Hệ Thống Dự Báo Nhu Cầu & Phân Tích Bán Lẻ")
st.markdown("Sử dụng mô hình **XGBoost** thông minh giúp tối ưu hóa hàng tồn kho và chiến lược định giá.")
st.divider()

# Tạo 2 Tab chức năng chính
tab_predict, tab_analytics = st.tabs(["🚀 Dự Báo Nhu Cầu (Prediction)", "📊 Phân Tích Dữ Liệu (Analytics)"])

# ==================== TAB 1: DỰ BÁO NHU CẦU ====================
with tab_predict:
    st.subheader("Nhập thông số sản phẩm & thị trường")
    
    # Chia giao diện nhập liệu thành 2 cột cho gọn gàng, đẹp mắt
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.number_input("Giá bán hiện tại ($)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        discount = st.slider("Mức giảm giá (%)", min_value=0, max_value=100, value=10)
        inventory_level = st.number_input("Mức tồn kho hiện tại", min_value=0, max_value=100, value=20)
        
    with col2:
        promotion = st.selectbox("Chương trình khuyến mãi", options=[0, 1], format_func=lambda x: "Có khuyến mãi (1)" if x == 1 else "Không khuyến mãi (0)")
        competitor_pricing = st.number_input("Giá đối thủ cạnh tranh ($)", min_value=0.0, max_value=100.0, value=16.0, step=0.5)
        category = st.selectbox("Ngành hàng (Category)", label_encoders["Category"].classes_.tolist())

    st.divider()

    # Nút bấm dự đoán
    if st.button("🔮 Tiến hành dự báo Demand", type="primary", use_container_width=True):
        # Tạo DataFrame đầu vào với tên cột KHỚP 100% với lúc train
        input_data = pd.DataFrame({
            "Price": [price],
            "Discount": [discount],
            "Inventory Level": [inventory_level],
            "Promotion": [promotion],
            "Competitor Pricing": [competitor_pricing],
            "Category": [category]
        })

        # Mã hóa biến phân loại bằng LabelEncoder đã lưu
        input_encoded = input_data.copy()
        for col, encoder in label_encoders.items():
            if col in input_encoded.columns:
                input_encoded[col] = encoder.transform(input_encoded[col])

        # Dự đoán
        prediction = model.predict(input_encoded)
        predicted_value = int(np.round(prediction[0]))

        # Hiển thị kết quả nổi bật
        st.success(f"### 🎯 Nhu cầu dự kiến (Predicted Demand): **{predicted_value} đơn vị**")

        # --- PHẦN CHART BỔ SUNG NGAY SAU KHI PREDICT ---
        st.markdown("#### 📈 So sánh trực quan nhanh")
        fig, ax = plt.subplots(figsize=(8, 3))
        
        metrics_to_plot = {
            'Predicted Demand': predicted_value,
            'Inventory Level': inventory_level,
        }
        
        bars = ax.barh(list(metrics_to_plot.keys()), list(metrics_to_plot.values()), color=['#2b5c8f', '#4ca64c'])
        ax.set_xlabel("Số lượng (Units)")
        ax.set_title("Tương quan giữa Tồn kho và Nhu cầu dự báo")
        
        # Gắn giá trị lên đầu cột
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    va='center', ha='left', fontsize=10, fontweight='bold')
            
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)

# ==================== TAB 2: PHÂN TÍCH DỮ LIỆU (ANALYTICS) ====================
with tab_analytics:
    st.subheader("📊 Khám phá thông tin chi tiết từ tập dữ liệu")
    
    if df is not None:
        row1_col1, row1_col2 = st.columns(2)
        
        # 1. Biểu đồ Feature Importance của mô hình XGBoost
        with row1_col1:
            st.markdown("#### 🌟 Mức độ quan trọng của đặc trưng (Feature Importance)")
            feature_importance = pd.Series(
                model.feature_importances_,
                index=["Price", "Discount", "Inventory Level", "Promotion", "Competitor Pricing", "Category"]
            ).sort_values(ascending=True)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            feature_importance.plot(kind="barh", ax=ax, color="teal")
            ax.set_title("XGBoost Feature Importance")
            ax.set_xlabel("Độ quan trọng")
            st.pyplot(fig)

        # 2. Biểu đồ Demand theo Category
        with row1_col2:
            st.markdown("#### 🛍️ Nhu cầu trung bình theo Ngành hàng")
            fig, ax = plt.subplots(figsize=(6, 4))
            cat_demand = df.groupby("Category")["Demand"].mean().sort_values()
            cat_demand.plot(kind="barh", ax=ax, color="coral")
            ax.set_title("Average Demand by Category")
            ax.set_xlabel("Demand Trung Bình")
            st.pyplot(fig)
            
        st.divider()
        
        row2_col1, row2_col2 = st.columns(2)
        
        # 3. Biểu đồ phân tán Tồn kho vs Units Sold
        with row2_col1:
            st.markdown("#### 📦 Tồn kho vs Số lượng bán thực tế")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x="Inventory Level", y="Units Sold", ax=ax, alpha=0.6, color="purple")
            ax.set_title("Inventory vs Units Sold")
            st.pyplot(fig)

        # 4. Tác động của khuyến mãi (Promotion) đến Demand
        with row2_col2:
            st.markdown("#### 🏷️ Tác động của Khuyến mãi đến Nhu cầu")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df, x="Promotion", y="Demand", ax=ax, palette="Set2", errorbar=None)
            ax.set_title("Promotion Impact on Demand")
            ax.set_xlabel("Promotion (0: Không, 1: Có)")
            ax.set_ylabel("Demand Trung Bình")
            st.pyplot(fig)
    else:
        st.warning("Không tìm thấy file `demand_forecasting.csv` để hiển thị biểu đồ phân tích.")