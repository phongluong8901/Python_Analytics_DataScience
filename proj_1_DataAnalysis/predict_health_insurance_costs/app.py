import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Cấu hình trang Streamlit ---
st.set_page_config(
    page_title="Insurance Analytics & Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Tải các Artifacts đã lưu từ quá trình Train ---
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("scaler.pkl")
        le_gender = joblib.load("label_encoder_gender.pkl")
        le_diabetic = joblib.load("label_encoder_diabetic.pkl")
        le_smoker = joblib.load("label_encoder_smoker.pkl")
        model = joblib.load("best_model.pkl")
        df = pd.read_csv("insurance.csv")
        return scaler, le_gender, le_diabetic, le_smoker, model, df
    except Exception as e:
        st.error(f"Lỗi khi tải file artifacts hoặc dataset: {e}")
        return None, None, None, None, None, None

scaler, le_gender, le_diabetic, le_smoker, model, df = load_artifacts()

if model is not None:
    # --- 3. Sidebar điều hướng & Bộ lọc toàn cục ---
    st.sidebar.title("🧭 Điều hướng hệ thống")
    app_mode = st.sidebar.radio(
        "Chọn chức năng:",
        ["🔮 Dự báo chi phí cá nhân", "📊 Khám phá dữ liệu (EDA Dashboard)"]
    )

    # ==========================================
    # CHỨC NĂNG 1: DỰ BÁO CHI PHÍ BẢO HIỂM
    # ==========================================
    if app_mode == "🔮 Dự báo chi phí cá nhân":
        st.title("💡 Ứng dụng Dự báo Chi phí Bảo hiểm Y tế")
        st.write("Nhập thông tin hồ sơ bên dưới để hệ thống ước tính số tiền bảo hiểm chi trả dựa trên mô hình Machine Learning tối ưu.")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Tuổi (Age)", min_value=1, max_value=100, value=30)
                bmi = st.number_input("Chỉ số BMI", min_value=10.0, max_value=70.0, value=25.0)
                children = st.number_input("Số lượng con / phụ thuộc (Children)", min_value=0, max_value=10, value=0)

            with col2:
                bloodpressure = st.number_input("Huyết áp (Blood Pressure)", min_value=80, max_value=180, value=120)
                gender = st.selectbox("Giới tính (Gender)", options=le_gender.classes_)
                diabetic = st.selectbox("Tình trạng tiểu đường (Diabetic)", options=le_diabetic.classes_)
                smoker = st.selectbox("Thói quen hút thuốc (Smoker)", options=le_smoker.classes_)
            
            submitted = st.form_submit_button("🚀 Thực hiện dự báo chi phí")

        if submitted:
            input_data = pd.DataFrame({
                "age": [age],
                "gender": [gender],
                "bmi": [bmi],
                "bloodpressure": [bloodpressure],
                "diabetic": [diabetic],
                "children": [children],
                "smoker": [smoker]
            })

            input_data["gender"] = le_gender.transform(input_data["gender"])
            input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
            input_data["smoker"] = le_smoker.transform(input_data["smoker"])

            num_cols = ["age", "bmi", "bloodpressure", "children"]
            input_data[num_cols] = scaler.transform(input_data[num_cols])
            
            prediction = model.predict(input_data)

            st.markdown("---")
            st.subheader("🎯 Kết quả ước tính:")
            st.success(f"### Chi phí bảo hiểm dự kiến: **${prediction[0]:,.2f} USD**")
            
            if smoker.lower() == "yes" or smoker == "Yes":
                st.info("💡 **Gợi ý chuyên gia:** Thói quen hút thuốc đang làm tăng vọt chi phí bảo hiểm y tế của bạn.")

    # ==========================================
    # CHỨC NĂNG 2: EDA & DASHBOARD PHÂN TÍCH
    # ==========================================
    elif app_mode == "📊 Khám phá dữ liệu (EDA Dashboard)":
        st.title("📈 Insurance Data Analytics Dashboard")
        st.write("Bảng điều khiển trực quan tổng quan các chỉ số nhân khẩu học và yếu tố tác động đến chi phí bảo hiểm.")

        # --- Thanh bộ lọc phụ trên Sidebar cho Dashboard ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Bộ lọc Dashboard")
        selected_region = st.sidebar.multiselect(
            "Chọn khu vực (Region):",
            options=df["region"].unique(),
            default=df["region"].unique()
        )
        
        # Lọc dữ liệu theo sidebar
        filtered_df = df[df["region"].isin(selected_region)]

        # --- 1. HIỂN THỊ CÁC THẺ KPI TỔNG QUAN (METRICS) ---
        st.markdown("### 📌 Chỉ số tổng quan (KPIs)")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        total_patients = len(filtered_df)
        avg_claim = filtered_df["claim"].mean() if total_patients > 0 else 0
        avg_bmi = filtered_df["bmi"].mean() if total_patients > 0 else 0
        smoker_pct = (filtered_df["smoker"].str.lower() == "yes").mean() * 100 if total_patients > 0 else 0

        kpi1.metric("Tổng số hồ sơ", f"{total_patients:,}")
        kpi2.metric("Chi phí bảo hiểm TB", f"${avg_claim:,.2f}")
        kpi3.metric("Chỉ số BMI trung bình", f"{avg_bmi:.1f}")
        kpi4.metric("Tỷ lệ hút thuốc", f"{smoker_pct:.1f}%")

        st.markdown("---")

        # --- 2. BỐ CỤC DASHBOARD SONG SONG (CHARTS & TABLES) ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🔥 Mối quan hệ giữa BMI và Chi phí")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(data=filtered_df, x="bmi", y="claim", hue="smoker", alpha=0.7, ax=ax, palette="Set1")
            ax.set_title("BMI vs Claim theo tình trạng hút thuốc")
            st.pyplot(fig)

        with col_right:
            st.subheader("📊 Chi phí TB theo Khu vực & Hút thuốc")
            if not filtered_df.empty:
                pivot_table = pd.pivot_table(filtered_df, values="claim", index="region", columns="smoker", aggfunc="mean")
                fig, ax = plt.subplots(figsize=(7, 4.5))
                pivot_table.plot(kind="bar", ax=ax, colormap="Set2")
                ax.set_title("Trung bình Claim theo Khu vực")
                ax.set_ylabel("Mean Claim ($)")
                plt.xticks(rotation=0)
                st.pyplot(fig)
            else:
                st.warning("Không có dữ liệu phù hợp với bộ lọc.")

        st.markdown("---")

        # --- 3. HÀNG THỨ HAI CỦA DASHBOARD ---
        col_sub1, col_sub2 = st.columns(2)

        with col_sub1:
            st.subheader("📈 Phân phối biến số định lượng")
            numeric_cols = ["age", "bmi", "bloodpressure", "children", "claim"]
            selected_col = st.selectbox("Chọn biến số xem phân phối:", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(data=filtered_df, x=selected_col, kde=True, bins=25, ax=ax, color="teal")
            ax.set_title(f"Phân phối của {selected_col}")
            st.pyplot(fig)

        with col_sub2:
            st.subheader("📋 Bảng tổng hợp số liệu chi tiết")
            if not filtered_df.empty:
                summary_table = filtered_df.groupby("region").agg(
                    So_luong=("claim", "count"),
                    Chi_phi_TB=("claim", "mean"),
                    BMI_TB=("bmi", "mean"),
                    Tuoi_TB=("age", "mean")
                ).reset_index()
                summary_table.columns = ["Khu vực", "Số lượng", "Chi phí TB ($)", "BMI TB", "Tuổi TB"]
                st.dataframe(summary_table.style.format({
                    "Chi phí TB ($)": "${:,.2f}",
                    "BMI TB": "{:.1f}",
                    "Tuổi TB": "{:.1f}"
                }), use_container_width=True)
            else:
                st.warning("Không có dữ liệu.")