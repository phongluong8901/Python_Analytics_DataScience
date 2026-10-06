import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hotel Booking Analytics & Predictor",
    page_icon="🏨",
    layout="wide"
)

# Tải mô hình đã huấn luyện
@st.cache_resource
def load_model():
    return joblib.load("gb_booking_model.pkl")

# Tải dữ liệu mẫu để vẽ biểu đồ phân tích (đảm bảo file HotelData.xlsx cùng thư mục)
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("HotelData.xlsx")
        # Xử lý các cột phụ trợ giống như lúc train mô hình
        if "total_guests" not in df.columns:
            df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
        if "total_nights" not in df.columns:
            df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
        if "booking_status" in df.columns and df["booking_status"].dtype == object:
            df["booking_status_mapped"] = df["booking_status"].map({
                "Not_Canceled": 1,
                "Canceled": 0
            })
        return df
    except Exception as e:
        return None

model = load_model()
df = load_data()

# Tiêu đề ứng dụng
st.title("🏨 Hệ thống Phân tích & Dự đoán Hủy phòng Khách sạn")
st.markdown("Ứng dụng hỗ trợ khai thác dữ liệu (EDA), biểu đồ trực quan hóa và dự đoán khả năng hủy phòng dựa trên Machine Learning.")

# Tạo các Tab chức năng trên giao diện
tab1, tab2 = st.tabs(["🔮 Dự đoán Hủy phòng (Predict)", "📊 Phân tích Trực quan (EDA Charts)"])

# ================= TAB 1: DỰ ĐOÁN =================
with tab1:
    st.subheader("Nhập thông tin chi tiết đơn đặt phòng")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lead_time = st.slider("Lead Time (Số ngày đặt trước)", 0, 500, 50)
        avg_price = st.number_input("Average Price per Room (Giá phòng TB)", 0.0, 500.0, 100.0)
        special_requests = st.slider("Number of Special Requests (Số yêu cầu đặc biệt)", 0, 5, 1)
        
    with col2:
        total_guests = st.slider("Total Guests (Tổng số khách)", 1, 10, 2)
        total_nights = st.slider("Total Nights (Tổng số đêm lưu trú)", 1, 20, 3)
        repeated_guest = st.selectbox("Repeated Guest (Khách cũ hay mới)", [0, 1], format_func=lambda x: "Khách cũ (1)" if x == 1 else "Khách mới (0)")

    if st.button("🚀 Thực hiện Dự đoán", type="primary"):
        # Đóng gói dữ liệu đầu vào thành ma trận 2D đúng chuẩn mô hình yêu cầu
        input_data = np.array([[
            lead_time,
            avg_price,
            special_requests,
            total_guests,
            total_nights,
            repeated_guest
        ]])

        # Dự đoán nhãn và xác suất
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] # Xác suất tương ứng của nhãn (tùy thuộc vào cách map nhãn của bạn)

        st.markdown("---")
        st.subheader("📊 Kết quả dự đoán:")

        # Lưu ý: Mô hình GB của bạn dùng nhãn số (0 hoặc 1). 
        # Theo mapping chuẩn lúc huấn luyện: 1 = Not Canceled, 0 = Canceled
        if prediction == 0:
            st.error(f"🚨 **Khả năng cao khách sẽ HỦY PHÒNG (Canceled)**")
            st.warning(f"Xác suất rủi ro hủy phòng: **{(1 - prob) * 100:.2f}%**")
        else:
            st.success(f"✅ **Khách sẽ GIỮ PHÒNG (Not Canceled)**")
            st.info(f"Xác suất giữ phòng thành công: **{prob * 100:.2f}%**")

# ================= TAB 2: BIỂU ĐỒ PHÂN TÍCH (CHARTS) =================
with tab2:
    st.subheader("📈 Khám phá xu hướng dữ liệu lịch sử khách sạn")
    
    if df is not None:
        chart_type = st.selectbox(
            "Chọn biểu đồ phân tích muốn xem:",
            [
                "1. Phân phối trạng thái đặt phòng (Booking Status)",
                "2. Tỷ lệ hủy phòng theo Tháng đến (Arrival Month)",
                "3. So sánh Lead Time theo Trạng thái hủy phòng",
                "4. Biểu đồ tương quan các biến số (Correlation Heatmap)"
            ]
        )
        
        st.markdown("---")
        
        if "1." in chart_type:
            st.markdown("#### Biểu đồ đếm số lượng trạng thái đơn đặt phòng")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df, x="booking_status", palette="Set2", ax=ax)
            ax.set_title("Số lượng đơn đặt phòng theo trạng thái")
            ax.set_xlabel("Trạng thái")
            ax.set_ylabel("Số lượng")
            st.pyplot(fig)
            
        elif "2." in chart_type:
            st.markdown("#### Xu hướng số lượng booking theo tháng và trạng thái")
            if "arrival_month" in df.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.countplot(data=df, x="arrival_month", hue="booking_status", palette="viridis", ax=ax)
                ax.set_title("Biểu đồ phân bố booking theo tháng đến")
                ax.set_xlabel("Tháng đến")
                ax.set_ylabel("Số lượng")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Không tìm thấy cột 'arrival_month' trong DataFrame.")
                
        elif "3." in chart_type:
            st.markdown("#### Biểu đồ Boxplot: Lead Time và Trạng thái hủy phòng")
            if "lead_time" in df.columns and "booking_status" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df, x="booking_status", y="lead_time", palette="coolwarm", ax=ax)
                ax.set_title("So sánh Lead Time giữa các nhóm đơn phòng")
                ax.set_xlabel("Trạng thái đặt phòng")
                ax.set_ylabel("Số ngày đặt trước (Lead Time)")
                st.pyplot(fig)
            else:
                st.warning("Thiếu cột dữ liệu cần thiết để vẽ Boxplot.")
                
        elif "4." in chart_type:
            st.markdown("#### Ma trận tương quan giữa các biến dạng số")
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            if len(num_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = df[num_cols].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
                ax.set_title("Ma trận tương quan Pearson")
                st.pyplot(fig)
            else:
                st.warning("Không có đủ cột dữ liệu số để tính ma trận tương quan.")
    else:
        st.error("⚠️ Không tìm thấy tệp dữ liệu `HotelData.xlsx` trong thư mục làm việc. Vui lòng đưa file Excel vào cùng thư mục với `app.py` để hiển thị biểu đồ phân tích!")