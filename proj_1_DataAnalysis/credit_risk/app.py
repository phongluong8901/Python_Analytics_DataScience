# ==============================================================================
# IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import time

# ==============================================================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp giao diện (Thẻ card, khoảng cách, font chữ)
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stMetric {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HÀM TẢI (LOAD) CÁC ARTIFACTS, ENCODERS VÀ DATASET
# ==============================================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("extra_trees_credit_model.pkl")
        cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account"]
        encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in cat_cols}
        target_encoder = joblib.load("target_encoder.pkl")
        
        df = pd.read_csv("german_credit_data.csv")
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
            
        return model, encoders, target_encoder, df
    except Exception as e:
        return None, None, None, None

model, encoders, target_encoder, df = load_artifacts()

if model is None:
    st.error("⚠️ Không tìm thấy các file mô hình hoặc dữ liệu cần thiết (`extra_trees_credit_model.pkl`, `*_encoder.pkl`, `german_credit_data.csv`). Vui lòng kiểm tra lại thư mục hiện tại.")
else:
    # ==============================================================================
    # 3. THANH ĐIỀU HƯỚNG (SIDEBAR NAVIGATION CHUYÊN NGHIỆP)
    # ==============================================================================
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/bank-building.png", width=70)
        st.markdown("### **CREDIT RISK AI**")
        st.caption("Hệ thống chấm điểm tín dụng thông minh ứng dụng Machine Learning.")
        st.markdown("---")
        
        app_mode = st.radio(
            "CHUYỂN TRANG:",
            ["🔮 Mô hình Dự báo Khoản vay", "📊 Phân tích Danh mục & EDA"]
        )
        
        st.markdown("---")
        st.markdown("**Thông tin hệ thống:**")
        st.info("• **Model:** Extra Trees Classifier\n• **Dataset:** German Credit Risk\n• **Trạng thái:** Hoạt động ổn định 🟢")

    # ==============================================================================
    # CHỨC NĂNG 1: DỰ BÁO RỦI RO TÍN DỤNG CÁ NHÂN
    # ==============================================================================
    if app_mode == "🔮 Mô hình Dự báo Khoản vay":
        st.title("🔮 Chấm điểm & Đánh giá Rủi ro Tín dụng Tự động")
        st.markdown("Điền thông tin hồ sơ tín dụng của khách hàng bên dưới để hệ thống tiến hành phân tích dự báo thời gian thực.")
        
        with st.form("prediction_form"):
            st.markdown("#### 📝 Thông tin hồ sơ khách hàng")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Tuổi (Age)", min_value=18, max_value=100, value=32, help="Tuổi của khách hàng")
                sex = st.selectbox("Giới tính (Sex)", options=encoders["Sex"].classes_)
            
            with col2:
                job = st.selectbox("Nghề nghiệp / Trình độ công việc (Job Index)", options=[0, 1, 2, 3], format_func=lambda x: f"Cấp độ {x}")
                housing = st.selectbox("Hình thức nhà ở (Housing)", options=encoders["Housing"].classes_)
                
            with col3:
                saving_accounts = st.selectbox("Số dư Tiết kiệm (Saving accounts)", options=encoders["Saving accounts"].classes_)
                checking_account = st.selectbox("Số dư Tài khoản thanh toán (Checking account)", options=encoders["Checking account"].classes_)

            st.markdown("---")
            st.markdown("#### 💰 Thông tin khoản đề xuất vay")
            col_loan1, col_loan2 = st.columns(2)
            with col_loan1:
                credit_amount = st.number_input("Số tiền vay đề xuất ($)", min_value=100, max_value=100000, value=2500, step=100)
            with col_loan2:
                duration = st.number_input("Thời hạn vay (Tháng)", min_value=1, max_value=72, value=12)

            st.markdown("")
            submitted = st.form_submit_button("🚀 Thực hiện Chấm điểm Tín dụng", use_container_width=True)

        if submitted:
            # Hiệu ứng loading giả lập chuyên nghiệp
            with st.spinner("Đang trích xuất đặc trưng và chạy mô hình phân loại..."):
                time.sleep(0.8) # Tạo cảm giác xử lý hệ thống thật
                
                # Mã hóa dữ liệu đầu vào
                sex_encoded = encoders["Sex"].transform([sex])[0]
                housing_encoded = encoders["Housing"].transform([housing])[0]
                saving_encoded = encoders["Saving accounts"].transform([saving_accounts])[0]
                checking_encoded = encoders["Checking account"].transform([checking_account])[0]

                input_df = pd.DataFrame({
                    "Sex": [sex_encoded],
                    "Job": [job],
                    "Housing": [housing_encoded],
                    "Saving accounts": [saving_encoded],
                    "Checking account": [checking_encoded],
                    "Credit amount": [credit_amount],
                    "Duration": [duration]
                })

                pred = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0]

            st.markdown("---")
            st.subheader("📊 Kết quả Đánh giá Rủi ro")
            
            result_label = target_encoder.inverse_transform([pred])[0]
            confidence = pred_proba[pred] * 100

            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if result_label.lower() == "good" or pred == 1:
                    st.success("### **PHÂN LOẠI: TỐT (GOOD)**")
                    st.metric(label="Mức độ tin cậy mô hình", value=f"{confidence:.2f}%", delta="Rủi ro thấp")
                else:
                    st.error("### **PHÂN LOẠI: XẤU (BAD)**")
                    st.metric(label="Mức độ tin cậy mô hình", value=f"{confidence:.2f}%", delta="Rủi ro cao", delta_color="inverse")
            
            with res_col2:
                st.markdown("**Khuyến nghị quyết định từ hệ thống:**")
                if result_label.lower() == "good" or pred == 1:
                    st.markdown("""
                    - ✅ **Đề xuất:** **Phê duyệt khoản vay**.
                    - Khách hàng có cấu trúc tài chính ổn định và mức độ tín nhiệm cao theo thuật toán phân loại.
                    - Có thể tiến hành giải ngân theo tiêu chuẩn thông thường.
                    """)
                else:
                    st.markdown("""
                    - ❌ **Đề xuất:** **Từ chối / Yêu cầu tài sản đảm bảo bổ sung**.
                    - Khách hàng có nguy cơ vỡ nợ cao dựa trên các biến lịch sử tài khoản tiết kiệm và thời hạn vay.
                    - Cần thẩm định thêm thu nhập thực tế trước khi ra quyết định cuối cùng.
                    """)

    # ==============================================================================
    # CHỨC NĂNG 2: EDA & DASHBOARD PHÂN TÍCH DỮ LIỆU
    # ==============================================================================
    elif app_mode == "📊 Phân tích Danh mục & EDA":
        st.title("📊 Tổng quan Dữ liệu & Phân tích Rủi ro Danh mục")
        st.markdown("Theo dõi các chỉ số quan trọng, xu hướng phân phối tài chính và cấu trúc danh mục tín dụng toàn cục.")

        # --- Bộ lọc nâng cao trên Sidebar ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Bộ lọc dữ liệu trực quan")
        
        selected_sex = st.sidebar.multiselect(
            "Lọc theo Giới tính:",
            options=df["Sex"].unique(),
            default=df["Sex"].unique()
        )
        
        max_credit = int(df["Credit amount"].max())
        credit_range = st.sidebar.slider(
            "Khoảng số tiền vay ($):",
            min_value=int(df["Credit amount"].min()),
            max_value=max_credit,
            value=(int(df["Credit amount"].min()), max_credit)
        )

        filtered_df = df[
            (df["Sex"].isin(selected_sex)) & 
            (df["Credit amount"] >= credit_range[0]) & 
            (df["Credit amount"] <= credit_range[1])
        ]

        # --- Thẻ KPI số liệu cấp cao ---
        st.markdown("### 📌 Chỉ số Hiệu suất Chính (KPIs)")
        k1, k2, k3, k4 = st.columns(4)

        total_records = len(filtered_df)
        mean_amount = filtered_df["Credit amount"].mean() if total_records > 0 else 0
        mean_duration = filtered_df["Duration"].mean() if total_records > 0 else 0
        good_rate = (filtered_df["Risk"] == "good").mean() * 100 if "good" in filtered_df["Risk"].values else 0

        k1.metric("Tổng hồ sơ lọc", f"{total_records:,}", f"/{len(df):,} tổng")
        k2.metric("Khoản vay trung bình", f"${mean_amount:,.0f}")
        k3.metric("Thời hạn trung bình", f"{mean_duration:.1f} tháng")
        k4.metric("Tỷ lệ tín dụng đạt chuẩn", f"{good_rate:.1f}%")

        st.markdown("---")

        # --- Biểu đồ hàng 1 ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("📦 Phân phối số tiền vay theo Nhóm Rủi ro")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.boxplot(data=filtered_df, x="Risk", y="Credit amount", hue="Risk", palette="Set2", legend=False, ax=ax)
            ax.set_title("Credit Amount Distribution by Risk Class")
            ax.set_xlabel("Mức độ rủi ro (Risk)")
            ax.set_ylabel("Số tiền vay ($)")
            st.pyplot(fig)

        with chart_col2:
            st.subheader("🔥 Bản đồ Tương quan Biến định lượng (Correlation)")
            num_cols = filtered_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 1:
                corr = filtered_df[num_cols].corr()
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
                ax.set_title("Pearson Correlation Matrix")
                st.pyplot(fig)

        st.markdown("---")

        # --- Biểu đồ hàng 2 ---
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.subheader("⏳ Biểu đồ Mật độ Thời hạn Vay (Duration Density)")
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(data=filtered_df, x="Duration", kde=True, bins=20, color="darkcyan", ax=ax)
            ax.set_title("Duration Histogram & KDE")
            ax.set_xlabel("Thời hạn (tháng)")
            ax.set_ylabel("Tần suất")
            st.pyplot(fig)

        with chart_col4:
            st.subheader("📋 Bảng thống kê chi tiết theo Mục đích vay")
            if "Purpose" in filtered_df.columns and not filtered_df.empty:
                agg_df = filtered_df.groupby("Purpose").agg(
                    So_luong=("Credit amount", "count"),
                    Tong_tien_vay=("Credit amount", "sum"),
                    So_tien_TB=("Credit amount", "mean")
                ).reset_index()
                agg_df.columns = ["Mục đích", "Số lượng", "Tổng dư nợ ($)", "Trung bình ($)"]
                
                st.dataframe(
                    agg_df.style.format({
                        "Tổng dư nợ ($)": "${:,.0f}",
                        "Trung bình ($)": "${:,.0f}"
                    }),
                    use_container_width=True,
                    height=200
                )
            else:
                st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")