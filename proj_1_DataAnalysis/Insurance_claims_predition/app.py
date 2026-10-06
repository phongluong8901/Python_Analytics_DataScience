import joblib
import pandas as pd
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# 1. Load model và các file cấu hình
model = joblib.load("claim_model.pkl")
model_columns = joblib.load("model_columns.pkl")
model_options = joblib.load("model_options.pkl")

# 2. Cấu hình giao diện trang (Mở rộng layout 'wide')
st.set_page_config(
    page_title="Insurance Claim Risk Dashboard", page_icon="🛡️", layout="wide"
)

# 3. Tiêu đề chính
st.title("🛡️ Insurance Claim & Fraud Risk Analytics Dashboard")
st.markdown(
    "Hệ thống thẩm định nâng cao tích hợp mô hình Machine Learning, cung cấp"
    " các chỉ số phân tích rủi ro và giải thích quyết định theo thời gian"
    " thực."
)

# Thêm banner hiển thị độ chính xác tổng thể của mô hình
st.info(
    "💡 **Thông tin & Hiệu suất Mô hình:** Hệ thống đang sử dụng thuật toán"
    " **Random Forest** (Đã tối ưu cho dữ liệu mất cân bằng). Chỉ số đánh giá"
    " trên tập kiểm thử: **ROC-AUC = 0.6537** | **Khả năng bắt gian lận (Recall"
    " Class 1) = 60.0%**."
)

st.divider()

# 4. Chia bố cục thành 2 cột: Cột trái (Nhập liệu), Cột phải (Kết quả & Phân tích chuyên sâu)
left_col, right_col = st.columns([1.2, 1.8], gap="large")

with left_col:
  st.subheader("📝 Thông tin Hợp đồng & Xe")

  with st.expander("👤 Thông tin Khách hàng & Hợp đồng", expanded=True):
    subscription_length = st.slider(
        "Thời gian tham gia bảo hiểm (năm)", 0.0, 14.0, 5.7, step=0.1
    )
    customer_age = st.slider("Tuổi khách hàng", 18, 85, 44)
    region_density = st.number_input(
        "Mật độ dân số khu vực", min_value=0, max_value=80000, value=8794, step=100
    )

  with st.expander("🚗 Thông số Xe & Trang bị an toàn", expanded=True):
    vehicle_model = st.selectbox("Dòng xe (Vehicle model)", model_options)
    vehicle_age = st.slider("Tuổi của xe (năm)", 0.0, 20.0, 1.2, step=0.1)

    st.markdown("**Các tính năng an toàn trang bị trên xe:**")
    col_a, col_b = st.columns(2)
    with col_a:
      is_parking_sensors = st.checkbox("Cảm biến lùi", value=True)
      is_front_fog_lights = st.checkbox("Đèn sương mù trước", value=True)
      is_brake_assist = st.checkbox("Hỗ trợ phanh khẩn cấp", value=True)
      is_power_steering = st.checkbox("Trợ lực lái", value=True)
    with col_b:
      is_driver_seat_height_adjustable = st.checkbox(
          "Chỉnh độ cao ghế lái", value=True
      )
      is_day_night_rear_view_mirror = st.checkbox(
          "Gương chiếu hậu ngày/đêm", value=True
      )
      is_speed_alert = st.checkbox("Cảnh báo tốc độ", value=True)

  predict_btn = st.button(
      "🚀 Chạy Phân tích & Đánh giá Rủi ro",
      type="primary",
      use_container_width=True,
  )

with right_col:
  st.subheader("📊 Kết quả Phân tích & Thẩm định Chuyên sâu")

  if predict_btn:
    # Đóng gói dữ liệu đầu vào
    input_dict = {
        "subscription_length": subscription_length,
        "vehicle_age": vehicle_age,
        "customer_age": customer_age,
        "region_density": region_density,
        "is_parking_sensors": int(is_parking_sensors),
        "is_front_fog_lights": int(is_front_fog_lights),
        "is_brake_assist": int(is_brake_assist),
        "is_power_steering": int(is_power_steering),
        "is_driver_seat_height_adjustable": int(
            is_driver_seat_height_adjustable
        ),
        "is_day_night_rear_view_mirror": int(is_day_night_rear_view_mirror),
        "is_speed_alert": int(is_speed_alert),
    }

    input_df = pd.DataFrame([input_dict])

    # Xử lý One-Hot Encoding cho model xe
    for col in model_columns:
      if col.startswith("model_"):
        input_df[col] = 1 if col == f"model_{vehicle_model}" else 0

    # Khớp chuẩn danh sách cột
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Dự đoán xác suất
    proba_all = model.predict_proba(input_df)[0]
    proba_safe = proba_all[0]
    proba_fraud = proba_all[1]
    prediction = model.predict(input_df)[0]

    # Tính độ tự tin của mô hình
    confidence = abs(proba_fraud - 0.5) * 200

    # Hiển thị các chỉ số cốt lõi qua Metric Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
      st.metric(
          label="Xác suất Rủi ro (Class 1)", value=f"{proba_fraud * 100:.2f}%"
      )
    with m_col2:
      risk_level = (
          "RẤT CAO"
          if proba_fraud > 0.6
          else ("TRUNG BÌNH" if proba_fraud > 0.3 else "THẤP")
      )
      st.metric(label="Mức độ Cảnh báo", value=risk_level)
    with m_col3:
      st.metric(label="Độ tin cậy dự đoán", value=f"{confidence:.1f}%")

    # Thanh tiến trình rủi ro trực quan
    st.markdown("---")
    st.write("**Thanh đo mức độ rủi ro hệ thống:**")
    st.progress(float(min(max(proba_fraud, 0.0), 1.0)))

    # Phân phối xác suất chi tiết 2 lớp
    st.markdown("### 📈 Phân phối Xác suất Dự đoán")
    prob_df = pd.DataFrame({
        "Trạng thái": ["Hồ sơ An toàn (0)", "Hồ sơ Rủi ro / Gian lận (1)"],
        "Xác suất (%)": [
            round(proba_safe * 100, 2),
            round(proba_fraud * 100, 2),
        ],
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Khuyến nghị nghiệp vụ tự động
    if prediction == 1 or proba_fraud > 0.5:
      st.error(
          "⚠️ **CẢNH BÁO: HỒ SƠ NÀY CÓ KHẢ NĂNG KHIẾU NẠI / GIAN LẬN CAO!**"
          "\n\n* **Đề xuất hành động:** Tạm hoãn phê duyệt tự động. Chuyển hồ"
          " sơ sang bộ phận điều tra chuyên sâu để xác minh lịch sử bảo"
          " dưỡng, thông số xe và giấy tờ liên quan."
      )
    else:
      st.success(
          "✅ **HỒ SƠ ĐẠT TIÊU CHUẨN AN TOÀN**"
          "\n\n* **Đề xuất hành động:** Hồ sơ hoàn toàn hợp lệ, có thể tiến"
          " hành cấp đơn hoặc duyệt bồi thường nhanh chóng."
      )

    # Phân tích tác động đặc trưng
    st.markdown("### 🔍 Phân tích Tác động Đặc trưng (Feature Importance)")
    try:
      importances = pd.Series(
          model.feature_importances_, index=model_columns
      ).sort_values(ascending=False)
      top_features = importances.head(5)
      st.write(
          "Các yếu tố trọng yếu chi phối kết quả dự đoán của mô hình trên hồ"
          " sơ này:"
      )
      st.bar_chart(top_features)
    except Exception:
      st.info(
          "Mô hình hiện tại không hỗ trợ trích xuất biểu đồ phân tích thành"
          " phần."
      )

    # Bảng tóm tắt thông số đầu vào
    with st.expander("📋 Xem lại Bảng thông số đầu vào đã xử lý"):
      st.dataframe(input_df, use_container_width=True)

  else:
    st.info(
        "👉 Vui lòng cấu hình các thông số ở menu bên trái và bấm nút **'Chạy"
        " Phân tích & Đánh giá Rủi ro'** để hiển thị kết quả thẩm định."
    )
    st.image(
        "https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&w=900&q=80",
        caption="Dashboard Thẩm định Bảo hiểm Tự động",
        use_container_width=True,
    )