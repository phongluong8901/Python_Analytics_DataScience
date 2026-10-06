import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cấu hình trang Streamlit (Giao diện rộng, tiêu đề chuyên nghiệp)
st.set_page_config(
    page_title="Telecom Churn Intelligence App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tải model và scaler đã được lưu từ quá trình huấn luyện (Sử dụng cache để tối ưu hiệu suất)
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_artifacts()

# Giao diện Sidebar điều hướng chính
st.sidebar.title("🧭 Điều Hướng Hệ Thống")
app_mode = st.sidebar.selectbox("Chọn chức năng", ["🔮 Dự Đoán Khách Hàng", "📊 Dashboard Phân Tích"])

if app_mode == "🔮 Dự Đoán Khách Hàng":
    st.title("🔮 Hệ Thống Dự Đoán Khách Hàng Rời Bỏ (Telecom Churn)")
    st.markdown("Nhập thông tin chi tiết của khách hàng bên dưới để hệ thống phân tích nguy cơ rời bỏ dịch vụ dựa trên mô hình Machine Learning đã huấn luyện.")
    st.divider()

    # Sử dụng form để gom cụm thao tác nhập liệu chuyên nghiệp hơn
    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Độ tuổi khách hàng (Age)", min_value=10, max_value=100, value=30, step=1)
            gender = st.selectbox("Giới tính (Gender)", ["Male", "Female"])
            
        with col2:
            tenure = st.slider("Số tháng gắn bó (Tenure)", min_value=0, max_value=100, value=12, step=1)
            monthlycharges = st.slider("Cước phí hàng tháng ($)", min_value=0, max_value=200, value=75, step=1)
            
        st.markdown("")
        submit_button = st.form_submit_button(label="🚀 Chạy Dự Đoán Ngay", use_container_width=True)

    # Xử lý logic khi người dùng nhấn nút dự đoán
    if submit_button:
        # Chuyển đổi giới tính khớp với quy trình train (Female: 1, Male: 0)
        gender_selected = 1 if gender == "Female" else 0
        
        # Gom cụm dữ liệu và đưa qua scaler
        X_raw = [age, gender_selected, tenure, monthlycharges]
        X_array = scaler.transform([np.array(X_raw)])
        
        # Dự đoán kết quả từ model
        prediction = model.predict(X_array)[0]
        
        st.divider()
        st.subheader("📋 Kết Quả Phân Tích Chi Tiết")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if prediction == 1:
                st.error("⚠️ CẢNH BÁO: Khách hàng có nguy cơ RỜI BỎ (Churn)")
            else:
                st.success("✅ AN TOÀN: Khách hàng tiếp tục ở lại (Not Churn)")
                
        with res_col2:
            st.info(f"**Mã kết quả dự đoán:** `{prediction}` (1: Churn | 0: Not Churn)\n\n"
                    f"*Thông số đầu vào:* Tuổi `{age}`, Giới tính `{gender}`, Thời gian gắn bó `{tenure}` tháng, Cước phí `${monthlycharges}`/tháng.")

else:
    st.title("📊 Dashboard Phân Tích & Thống Kê Tổng Quan")
    st.markdown("Trực quan hóa các chỉ số quan trọng và hành vi của khách hàng viễn thông.")
    st.divider()

    # Hàng 1: Các chỉ số KPI tổng quan
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Tổng số khách hàng", value="7,043", delta="+124 tháng này")
    kpi2.metric(label="Tỷ lệ rời bỏ (Churn Rate)", value="26.5%", delta="-1.2%")
    kpi3.metric(label="Cước phí trung bình", value="$64.76", delta="+$2.1")
    kpi4.metric(label="Thời gian gắn bó TB", value="32.4 tháng", delta="+1.5 tháng")

    st.divider()
    
    # Hàng 2: Các biểu đồ phân tích trực quan
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Cước phí theo hình thức hợp đồng")
        chart_data = pd.DataFrame(
            {
                "Month-to-Month": [75, 80, 85, 90, 70],
                "One Year": [60, 62, 58, 65, 63],
                "Two Year": [45, 48, 50, 42, 46]
            }
        )
        st.bar_chart(chart_data)
        
    with col_chart2:
        st.subheader("📉 Xu hướng Churn theo thời gian gắn bó")
        tenure_chart = pd.DataFrame({
            "Nhóm gắn bó (Tháng)": ["0-12", "13-24", "25-36", "37-48", "49-60", "61+"],
            "Tỷ lệ Churn (%)": [45, 28, 18, 12, 8, 5]
        }).set_index("Nhóm gắn bó (Tháng)")
        st.line_chart(tenure_chart)