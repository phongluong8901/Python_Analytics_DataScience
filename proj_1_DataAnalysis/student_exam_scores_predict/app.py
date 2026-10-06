import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình trang (Phải đặt ở dòng đầu tiên của Streamlit)
st.set_page_config(
    page_title="Student Performance Analytics & Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tải mô hình đã train (hoặc xử lý ngoại lệ nếu chưa có file)
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_model.pkl")
    except:
        return None

model = load_model()

# Tải tập dữ liệu mẫu để làm phần Data Analytics Dashboard (giả định bạn có file data.csv, nếu không sẽ tự sinh dữ liệu demo)
@st.cache_data
def load_data():
    try:
        return pd.read_csv("student_data.csv")
    except:
        # Tạo dữ liệu giả lập minh họa nếu chưa có file csv gốc
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "study_hours": np.random.uniform(1, 12, n),
            "attendance": np.random.uniform(50, 100, n),
            "mental_health": np.random.randint(1, 11, n),
            "sleep_hours": np.random.uniform(4, 10, n),
            "part_time_job": np.random.choice(["No", "Yes"], n, p=[0.7, 0.3]),
            "exam_score": np.random.uniform(40, 100, n)
        })

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎓 Student System")
app_mode = st.sidebar.radio("Chọn chức năng:", ["🔮 Dự đoán Điểm số (Predictor)", "📊 Phân tích Dữ liệu (Analytics Dashboard)"])

st.sidebar.markdown("---")
st.sidebar.info("Hệ thống hỗ trợ học tập thông minh tích hợp Machine Learning và EDA.")

# ==================== CHỨC NĂNG 1: PREDICTOR ====================
if app_mode == "🔮 Dự đoán Điểm số (Predictor)":
    st.title("🎓 Dự đoán Kết quả Học tập Sinh viên")
    st.markdown("Nhập các thông số thói quen học tập và lối sống của bạn bên dưới để hệ thống dự đoán điểm thi chính xác.")

    if model is None:
        st.error("⚠️ Không tìm thấy file `best_model.pkl`. Hãy chắc chắn bạn đã chạy xong script huấn luyện và lưu mô hình.")
    else:
        # Chia bố cục thành 2 cột cho gọn gàng, chuyên nghiệp
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("📝 Thông số đầu vào")
            study_hours = st.slider("Số giờ học mỗi ngày (Study Hours)", 0.0, 24.0, 5.0, 0.5)
            attendance = st.slider("Tỷ lệ đến lớp (%)", 0.0, 100.0, 85.0, 1.0)
            mental_health = st.slider("Chỉ số sức khỏe tinh thần (1-10)", 1.0, 10.0, 7.0, 1.0)
            sleep_hours = st.slider("Số giờ ngủ mỗi đêm", 0.0, 12.0, 7.0, 0.5)
            part_time_job = st.selectbox("Có làm thêm ngoài giờ không?", ["No", "Yes"])

            ptj_encoded = 1 if part_time_job == "Yes" else 0

        with col2:
            st.subheader("🎯 Kết quả dự đoán")
            
            # Tạo hiệu ứng khoảng trống hoặc khung kết quả đẹp mắt
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Tiến hành Dự đoán", use_container_width=True, type="primary"):
                input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
                prediction = model.predict(input_data)[0]
                prediction = max(0, min(100, prediction)) # Giới hạn trong khoảng 0 - 100
                
                # Hiển thị dạng Metric card hoành tráng
                st.metric(label="Điểm thi dự kiến (Exam Score)", value=f"{prediction:.2f} / 100")
                
                # Đánh giá trực quan bằng thanh tiến trình (Progress bar)
                st.progress(int(prediction))
                
                if prediction >= 80:
                    st.success("🎉 Xuất sắc! Bạn đang duy trì phong độ học tập rất tuyệt vời.")
                elif prediction >= 50:
                    st.warning("⚠️ Khá ổn, nhưng bạn có thể tăng số giờ học hoặc cải thiện tỷ lệ chuyên cần để đạt điểm cao hơn.")
                else:
                    st.error("🚨 Cần cải thiện gấp! Điểm số dự kiến đang ở mức thấp, hãy tăng thời gian ôn tập và ngủ đủ giấc.")

# ==================== CHỨC NĂNG 2: DATA ANALYTICS DASHBOARD ====================
elif app_mode == "📊 Phân tích Dữ liệu (Analytics Dashboard)":
    st.title("📊 Dashboard Phân tích Dữ liệu Học tập")
    st.markdown("Khám phá các biểu đồ thống kê, xu hướng và ma trận tương quan từ tập dữ liệu sinh viên.")

    # Hiển thị các chỉ số tổng quan (KPI metrics)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Tổng số sinh viên", len(df))
    with kpi2:
        st.metric("Điểm TB dự kiến", f"{df['exam_score'].mean():.1f}")
    with kpi3:
        st.metric("Giờ học TB/ngày", f"{df['study_hours'].mean():.1f}h")
    with kpi4:
        st.metric("Chuyên cần TB", f"{df['attendance'].mean():.1f}%")

    st.markdown("---")

    # Hàng biểu đồ 1
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🔥 Ma trận tương quan (Correlation Matrix)")
        fig, ax = plt.subplots(figsize=(6, 4))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True)
        st.pyplot(fig)

    with c2:
        st.subheader("📈 Quan hệ giữa Giờ học và Điểm số")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="study_hours", y="exam_score", hue="part_time_job", palette="Set1", ax=ax)
        ax.set_title("Study Hours vs Exam Score")
        st.pyplot(fig)

    # Hàng biểu đồ 2
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("💤 Phân phối Số giờ ngủ")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["sleep_hours"], kde=True, color="teal", ax=ax)
        st.pyplot(fig)

    with c4:
        st.subheader("📊 Tác động của Việc làm thêm đến Điểm số")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="part_time_job", y="exam_score", palette="Set2", ax=ax)
        st.pyplot(fig)