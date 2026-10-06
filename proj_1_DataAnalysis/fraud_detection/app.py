import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình giao diện trang
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tải mô hình pipeline đã train
@st.cache_resource
def load_model():
    try:
        return joblib.load("fraud_detection_pipeline.pkl")
    except:
        return None

pipeline = load_model()

# Tải tập dữ liệu mẫu để chạy Analytics Dashboard
@st.cache_data
def load_data():
    try:
        return pd.read_csv("AIML Dataset.csv")
    except:
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "type": np.random.choice(["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"], n),
            "amount": np.random.exponential(50000, n),
            "oldbalanceOrg": np.random.exponential(100000, n),
            "newbalanceOrg": np.random.exponential(80000, n),
            "oldbalanceDest": np.random.exponential(50000, n),
            "newbalanceDest": np.random.exponential(90000, n),
            "isFraud": np.random.choice([0, 1], n, p=[0.98, 0.02])
        })

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ Hệ thống Giám sát")
app_mode = st.sidebar.radio("Chọn chức năng:", ["🚨 Dự đoán Giao dịch (Predictor)", "📊 Phân tích Dữ liệu Gian lận (Analytics)"])

st.sidebar.markdown("---")
st.sidebar.info("Ứng dụng tích hợp Machine Learning Pipeline phát hiện gian lận tài chính.")

# ==================== CHỨC NĂNG 1: PREDICTOR ====================
if app_mode == "🚨 Dự đoán Giao dịch (Predictor)":
    st.title("🚨 Kiểm tra Giao dịch Gian lận (Fraud Detection)")
    st.markdown("Nhập thông tin chi tiết của giao dịch tài chính để hệ thống đánh giá.")

    if pipeline is None:
        st.error("⚠️ Không tìm thấy file `fraud_detection_pipeline.pkl`. Hãy chắc chắn bạn đã chạy xong script train và lưu pipeline.")
    else:
        col1, col2 = st.columns([1.2, 1], gap="large")

        with col1:
            st.subheader("💳 Thông tin Giao dịch")
            trans_type = st.selectbox("Loại giao dịch (type)", ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT', 'PAYMENT'])
            amount = st.number_input("Số tiền giao dịch (amount)", min_value=0.0, max_value=10_000_000.0, value=5000.0, step=100.0)
            
            st.markdown("---")
            st.subheader("💰 Biến động Số dư")
            oldbalanceOrg = st.number_input("Số dư cũ của Người gửi (oldbalanceOrg)", min_value=0.0, max_value=20_000_000.0, value=10000.0)
            newbalanceOrig = st.number_input("Số dư mới của Người gửi (newbalanceOrig)", min_value=0.0, max_value=20_000_000.0, value=5000.0)
            oldbalanceDest = st.number_input("Số dư cũ của Người nhận (oldbalanceDest)", min_value=0.0, max_value=20_000_000.0, value=0.0)
            newbalanceDest = st.number_input("Số dư mới của Người nhận (newbalanceDest)", min_value=0.0, max_value=20_000_000.0, value=5000.0)

        with col2:
            st.subheader("🎯 Kết quả Đánh giá Rủi ro")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔍 Phân tích Giao dịch", use_container_width=True, type="primary"):
                # DataFrame đầu vào khớp tuyệt đối với 6 cột gốc lúc train model
                input_data = pd.DataFrame({
                    "type": [trans_type],
                    "amount": [amount],
                    "oldbalanceOrg": [oldbalanceOrg],
                    "newbalanceOrig": [newbalanceOrig],
                    "oldbalanceDest": [oldbalanceDest],
                    "newbalanceDest": [newbalanceDest]
                })
                
                # Dự đoán kết quả
                prediction = pipeline.predict(input_data)[0]
                
                try:
                    proba = pipeline.predict_proba(input_data)[0][1]
                except:
                    proba = None

                st.markdown("---")
                if prediction == 1:
                    st.error("### 🚨 CẢNH BÁO: Giao dịch Gian lận (FRAUD)")
                    st.markdown("Hệ thống phát hiện dấu hiệu bất thường cao, rủi ro gian lận lớn!")
                    if proba is not None:
                        st.metric("Xác suất rủi ro gian lận", f"{proba*100:.2f}%")
                else:
                    st.success("### ✅ Giao dịch Hợp lệ (NORMAL)")
                    st.markdown("Giao dịch nằm trong ngưỡng an toàn.")
                    if proba is not None:
                        st.metric("Mức độ an toàn", f"{(1-proba)*100:.2f}%")

# ==================== CHỨC NĂNG 2: DATA ANALYTICS DASHBOARD ====================
elif app_mode == "📊 Phân tích Dữ liệu Gian lận (Analytics)":
    st.title("📊 Dashboard Thống kê & Phân tích Dữ liệu Gian lận")
    
    total_trans = len(df)
    fraud_trans = df["isFraud"].sum() if "isFraud" in df.columns else 0
    fraud_rate = (fraud_trans / total_trans) * 100 if total_trans > 0 else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Tổng số giao dịch phân tích", f"{total_trans:,}")
    with kpi2:
        st.metric("Số vụ gian lận phát hiện", f"{fraud_trans:,}")
    with kpi3:
        st.metric("Tỷ lệ gian lận (%)", f"{fraud_rate:.2f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Tần suất theo Loại giao dịch")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="type", palette="Set2", ax=ax)
        st.pyplot(fig)

    with c2:
        st.subheader("⚠️ Tỷ lệ gian lận theo Loại giao dịch")
        if "isFraud" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
            fraud_by_type.plot(kind="bar", color="salmon", ax=ax)
            st.pyplot(fig)