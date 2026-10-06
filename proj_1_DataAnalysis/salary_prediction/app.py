import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Dự đoán Lương Nhân Viên & Đánh giá Mô hình",
    page_icon="💼",
    layout="centered",
)


# Hàm tải mô hình
@st.cache_resource
def load_model():
    try:
        model = joblib.load("linearmodel.pkl")
        return model
    except FileNotFoundError:
        return None


model = load_model()

# Tiêu đề ứng dụng
st.title("💼 Ứng Dụng Dự Đoán Lương & Đánh Giá Mô Hình")
st.markdown(
    "Ứng dụng sử dụng mô hình **Linear Regression** để dự đoán lương hàng năm và hiển thị các chỉ số độ lỗi như **RMSE**."
)

st.markdown("---")

# Kiểm tra mô hình
if model is None:
    st.error(
        "⚠️ Không tìm thấy tệp mô hình `linearmodel.pkl`. Vui lòng đặt tệp mô hình cùng thư mục với `app.py`!"
    )
else:
    # --- PHẦN 1: DỰ ĐOÁN CHO NHÂN VIÊN MỚI ---
    st.subheader("🔮 1. Dự đoán Mức Lương cho Nhân Viên")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            years = st.number_input(
                "Số năm kinh nghiệm (Years)",
                min_value=0.0,
                max_value=40.0,
                value=2.0,
                step=0.5,
            )

        with col2:
            job_rate = st.number_input(
                "Đánh giá công việc (Job Rate)",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
            )

        submit_button = st.form_submit_button(
            label="🚀 Dự đoán Mức Lương", use_container_width=True
        )

    if submit_button:
        input_data = pd.DataFrame(
            [[years, job_rate]], columns=["Years", "Job Rate"]
        )
        prediction = model.predict(input_data)
        predicted_salary = prediction[0]

        st.success(
            f"💰 Mức lương hàng năm dự kiến: **${predicted_salary:,.2f}**"
        )

    st.markdown("---")

    # --- PHẦN 2: ĐÁNH GIÁ MÔ HÌH & HIỂN THỊ RMSE ---
    st.subheader("📈 2. Đánh giá độ chính xác mô hình (RMSE, MAE)")
    st.markdown(
        "Tải lên tệp dữ liệu kiểm thử định dạng **Excel (.xlsx)** có chứa các cột `Years`, `Job Rate` và `Annual Salary` thực tế để tính toán **RMSE**."
    )

    # Đổi type cho phép tải file xlsx
    uploaded_file = st.file_uploader(
        "Chọn tệp Excel dữ liệu test", type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            # Đọc file Excel bằng pd.read_excel
            test_df = pd.read_excel(uploaded_file)

            # Kiểm tra xem tệp có đủ cột cần thiết không
            required_cols = ["Years", "Job Rate", "Annual Salary"]
            if all(col in test_df.columns for col in required_cols):
                X_test = test_df[["Years", "Job Rate"]]
                y_true = test_df["Annual Salary"]

                # Dự đoán trên tập test
                y_pred = model.predict(X_test)

                # Tính toán các chỉ số
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)

                # Hiển thị kết quả đánh giá bằng các metric card của Streamlit
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("RMSE (Sai số căn bậc hai)", f"${rmse:,.2f}")
                col_m2.metric("MAE (Sai số tuyệt đối)", f"${mae:,.2f}")
                col_m3.metric("MSE (Sai số bình phương)", f"${mse:,.2f}")

                with st.expander("🔍 Xem dữ liệu và kết quả dự đoán test"):
                    test_df["Predicted Salary"] = y_pred
                    st.dataframe(test_df.head(10))
            else:
                st.error(
                    f"⚠️ Tệp Excel thiếu các cột bắt buộc. Yêu cầu phải có: `{required_cols}`"
                )
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi đọc tệp Excel: {e}")
    else:
        st.info(
            "💡 Mẹo: Bạn hãy tải file Excel (`.xlsx`) chứa dữ liệu kiểm tra ở trên để ứng dụng tự động tính toán RMSE thực tế cho bạn."
        )

# Chân trang
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>HR Analytics & Linear Regression App | Powered by Streamlit</p>",
    unsafe_allow_html=True,
)