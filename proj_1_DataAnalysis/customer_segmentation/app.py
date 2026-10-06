import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide"
)

# Load model và scaler
try:
  kmeans = joblib.load("kmeans_model.pkl")
  scaler = joblib.load("scaler.pkl")
except Exception as e:
  st.error(
      f"Không tìm thấy file mô hình (`kmeans_model.pkl` hoặc `scaler.pkl`)."
      f" Lỗi: {e}"
  )
  st.stop()

# Dữ liệu trung bình mẫu của 6 cụm (dựa trên cluster_summary thực tế)
cluster_mean_data = {
    0: {
        "Age": 45.5,
        "Income": 22312,
        "Total_Spending": 130,
        "NumWebPurchases": 4.5,
        "NumStorePurchases": 4.8,
        "NumWebVisitsMonth": 9.1,
    },
    1: {
        "Age": 48.2,
        "Income": 66700,
        "Total_Spending": 3098,
        "NumWebPurchases": 2.1,
        "NumStorePurchases": 3.1,
        "NumWebVisitsMonth": 7.0,
    },
    2: {
        "Age": 59.0,
        "Income": 79900,
        "Total_Spending": 8126,
        "NumWebPurchases": 7.9,
        "NumStorePurchases": 7.1,
        "NumWebVisitsMonth": 6.6,
    },
    3: {
        "Age": 68.9,
        "Income": 68700,
        "Total_Spending": 8149,
        "NumWebPurchases": 5.4,
        "NumStorePurchases": 11.0,
        "NumWebVisitsMonth": 3.8,
    },
    4: {
        "Age": 66.7,
        "Income": 57500,
        "Total_Spending": 151,
        "NumWebPurchases": 2.4,
        "NumStorePurchases": 3.9,
        "NumWebVisitsMonth": 5.5,
    },
    5: {
        "Age": 66.1,
        "Income": 28300,
        "Total_Spending": 131,
        "NumWebPurchases": 4.2,
        "NumStorePurchases": 5.8,
        "NumWebVisitsMonth": 2.3,
    },
}

# Tiêu đề ứng dụng
st.title("📊 Dashboard Phân Khúc & Phân Tích Hành Vi Khách Hàng")
st.markdown(
    "Nhập thông tin khách hàng để hệ thống dự báo cụm và trực quan hóa qua hệ"
    " thống 3 biểu đồ chuyên sâu."
)
st.markdown("---")

# Chia giao diện nhập liệu thành 2 cột
col1, col2 = st.columns(2)

with col1:
  st.subheader("📥 Thông tin Tài chính & Độ tuổi")
  age = st.number_input(
      "Độ tuổi (Age)", min_value=18, max_value=100, value=45, step=1
  )
  income = st.number_input(
      "Thu nhập hàng năm ($) (Income)",
      min_value=0,
      max_value=200000,
      value=60000,
      step=1000,
  )
  total_spending = st.number_input(
      "Tổng chi tiêu 2 năm ($) (Total_Spending)",
      min_value=0,
      max_value=50000,
      value=3000,
      step=100,
  )

with col2:
  st.subheader("🛒 Thông tin Hành vi Mua sắm")
  num_web_purchases = st.number_input(
      "Số lần mua qua Web (NumWebPurchases)",
      min_value=0,
      max_value=100,
      value=5,
      step=1,
  )
  num_store_purchases = st.number_input(
      "Số lần mua tại Cửa hàng (NumStorePurchases)",
      min_value=0,
      max_value=100,
      value=5,
      step=1,
  )
  num_web_visits = st.number_input(
      "Số lần ghé thăm Web/tháng (NumWebVisitsMonth)",
      min_value=0,
      max_value=100,
      value=5,
      step=1,
  )

st.markdown("")

# Từ điển chân dung
cluster_personas = {
    0: {
        "name": "Nhóm Khách hàng trẻ, thu nhập thấp, ít chi tiêu",
        "desc": (
            "Khách hàng trẻ tuổi, thu nhập thấp, tổng chi tiêu rất hạn chế"
            " nhưng rất hay lướt web xem sản phẩm."
        ),
        "strategy": (
            "🎯 **Chiến lược:** Đẩy mạnh sản phẩm giá rẻ, chương trình khuyến"
            " mại sinh viên, dùng retargeting nhẹ nhàng."
        ),
    },
    1: {
        "name": "Nhóm Khách hàng trung niên, thu nhập khá nhưng dè dặt",
        "desc": (
            "Khách hàng có thu nhập tốt nhưng tổng chi tiêu ở mức trung bình,"
            " cẩn trọng trong quyết định mua sắm."
        ),
        "strategy": (
            "🎯 **Chiến lược:** Gửi email marketing giới thiệu lợi ích sản"
            " phẩm, review chất lượng từ chuyên gia để xây dựng lòng tin."
        ),
    },
    2: {
        "name": "Nhóm Khách hàng VIP cao cấp, hiện đại",
        "desc": (
            "⭐ **Con gà đẻ trứng vàng:** Thu nhập rất cao và tổng chi tiêu"
            " cực kỳ lớn. Mua sắm mạnh qua cả Website lẫn Cửa hàng."
        ),
        "strategy": (
            "🎯 **Chiến lược:** Chăm sóc đặc biệt, cung cấp dịch vụ VIP support,"
            " miễn phí vận chuyển, tặng quà độc quyền."
        ),
    },
    3: {
        "name": "Nhóm Khách hàng lớn tuổi truyền thống (Mua offline cực mạnh)",
        "desc": (
            "Khách hàng lớn tuổi, tài chính vững vàng, trung thành tuyệt đối"
            " với việc mua sắm trực tiếp tại cửa hàng vật lý."
        ),
        "strategy": (
            "🎯 **Chiến lược:** Đầu tư trải nghiệm không gian cửa hàng sang"
            " trọng, nhân viên tư vấn tận tình."
        ),
    },
    4: {
        "name": "Nhóm Khách hàng lớn tuổi, thu nhập khá nhưng thắt chặt chi tiêu",
        "desc": (
            "Khách hàng lớn tuổi, thu nhập ổn định nhưng tổng chi tiêu rất"
            " thấp (lối sống tiết kiệm tối đa)."
        ),
        "strategy": (
            "🎯 **Chiến lược:** Thận trọng khi phân bổ chi phí tiếp thị, tập"
            " trung tối ưu nguồn lực cho Cụm 2 và Cụm 3."
        ),
    },
    5: {
        "name": "Nhóm Khách hàng bình dân truyền thống",
        "desc": (
            "Phân khúc bình dân lớn tuổi, thu nhập thấp, mua sắm dựa trên nhu"
            " cầu thiết yếu tại cửa hàng gần nhà."
        ),
        "strategy": (
            "🎯 **Chiến lược:** Áp dụng chương trình giảm giá trực tiếp tại"
            " cửa hàng, combo tiết kiệm."
        ),
    },
}

if st.button("🚀 Chạy Phân Tích & Vẽ Biểu Đồ", use_container_width=True):
  input_data = pd.DataFrame({
      "Age": [age],
      "Income": [income],
      "Total_Spending": [total_spending],
      "NumWebPurchases": [num_web_purchases],
      "NumStorePurchases": [num_store_purchases],
      "NumWebVisitsMonth": [num_web_visits],
  })

  input_scaled = scaler.transform(input_data)
  cluster = int(kmeans.predict(input_scaled)[0])

  st.markdown("---")
  st.success(f"### 🎉 Kết quả Dự báo: Khách hàng thuộc **Cluster {cluster}**")

  if cluster in cluster_personas:
    p = cluster_personas[cluster]
    st.info(f"**Chân dung:** {p['name']}\n\n{p['desc']}")
    st.warning(p["strategy"])

  st.markdown("---")
  st.header("📈 Hệ Thống Biểu Đồ Phân Tích Chuyên Sâu")

  mean_vals = cluster_mean_data[cluster]

  # --- BIỂU ĐỒ 1: So sánh trực tiếp khách hàng với mức chuẩn của Cụm ---
  st.subheader(
      f"1️⃣ Biểu đồ 1: Khách hàng hiện tại so với Trung bình Cluster {cluster}"
  )
  chart1_data = pd.DataFrame({
      "Chỉ số": [
          "Thu nhập (k$)",
          "Tổng chi tiêu ($)",
          "Mua Web",
          "Mua Cửa hàng",
          "Lướt Web/tháng",
      ],
      "Khách hàng này": [
          income / 1000,
          total_spending,
          num_web_purchases,
          num_store_purchases,
          num_web_visits,
      ],
      f"Trung bình Cluster {cluster}": [
          mean_vals["Income"] / 1000,
          mean_vals["Total_Spending"],
          mean_vals["NumWebPurchases"],
          mean_vals["NumStorePurchases"],
          mean_vals["NumWebVisitsMonth"],
      ],
  })
  chart1_data.set_index("Chỉ số", inplace=True)
  st.bar_chart(chart1_data)

  # Chia 2 cột để chứa biểu đồ 2 và biểu đồ 3 cho cân xứng giao diện
  chart_col1, chart_col2 = st.columns(2)

  # --- BIỂU ĐỒ 2: Tỷ trọng kênh mua sắm (Web vs Cửa hàng) ---
  with chart_col1:
    st.subheader("2️⃣ Biểu đồ 2: Cơ cấu Kênh mua sắm")
    channels_df = pd.DataFrame({
        "Kênh": ["Mua qua Web", "Mua tại Cửa hàng"],
        "Số lượng": [num_web_purchases, num_store_purchases],
    })
    channels_df.set_index("Kênh", inplace=True)
    st.bar_chart(channels_df, color="#FF4B4B")

  # --- BIỂU ĐỒ 3: Tương quan Thu nhập và Tổng chi tiêu so với toàn bộ các cụm trung bình ---
  with chart_col2:
    st.subheader("3️⃣ Biểu đồ 3: Vị thế Thu nhập các Cụm")
    all_clusters_income = {
        f"Cụm {k}": v["Income"] for k, v in cluster_mean_data.items()
    }
    income_df = pd.DataFrame(
        list(all_clusters_income.items()), columns=["Cluster", "Thu nhập TB"]
    )
    income_df.set_index("Cluster", inplace=True)

    # Dùng line_chart hoặc bar_chart thể hiện vị thế thu nhập các cụm
    st.bar_chart(income_df, color="#00C0F2")
    st.caption(
        "Giúp định vị mức thu nhập trung bình của cụm khách hàng này so với"
        " mặt bằng chung các cụm khác."
    )