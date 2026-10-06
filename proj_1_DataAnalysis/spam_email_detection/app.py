import streamlit as st
import pickle
import os

# Cấu hình giao diện trang web Streamlit
st.set_page_config(
    page_title="Spam Email Detection App",
    page_icon="📩",
    layout="wide"
)

# Hàm tải mô hình và bộ vectorizer từ ổ cứng lên
@st.cache_resource
def load_artifacts():
    model_path = "spam_classifier.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

model, vectorizer = load_artifacts()

# --- SIDEBAR: THÔNG TIN & HƯỚNG DẪN ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield-file.png", width=70)
    st.title("Bảng điều khiển")
    st.info("Ứng dụng sử dụng mô hình học máy kết hợp thuật toán **TF-IDF** để lọc tin nhắn rác tự động.")
    
    st.markdown("---")
    st.subheader("⚙️ Trạng thái hệ thống")
    if model is not None and vectorizer is not None:
        st.success("Mô hình: Đã sẵn sàng ✅")
    else:
        st.error("Mô hình: Chưa tìm thấy file ❌")
        
    st.markdown("---")
    st.markdown("### 💡 Hướng dẫn:")
    st.markdown("1. Nhập hoặc chọn nhanh mẫu văn bản.")
    st.markdown("2. Bấm nút **Phân tích nội dung**.")
    st.markdown("3. Xem kết quả dự đoán và biểu đồ xác suất.")

# --- GIAO DIỆN CHÍNH ---
st.title("📩 Hệ thống Phát hiện Thư rác (Spam Email Detection)")
st.markdown("Phân tích thông minh giúp phân biệt nhanh chóng giữa **Thư bình thường (Ham)** và **Thư rác (Spam)**.")
st.markdown("---")

if model is None or vectorizer is None:
    st.error("⚠️ Không tìm thấy file `spam_classifier.pkl` hoặc `tfidf_vectorizer.pkl`. Vui lòng đặt đúng thư mục chứa file mô hình!")
else:
    # Chia bố cục giao diện chính thành 2 cột (Cột trái: Nhập liệu & Mẫu nhanh, Cột phải: Kết quả & Biểu đồ)
    col_input, col_result = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.subheader("✍️ Nhập nội dung cần kiểm tra:")
        
        # Danh sách mẫu câu nhanh
        sample_texts = {
            "🎁 [Spam] Trúng thưởng": "Congratulations! You have won a FREE $1000 gift card. Click the link to claim your prize now!",
            "🚨 [Spam] Cảnh báo tài khoản": "URGENT! Your bank account has been locked. Verify your credentials immediately at http://fake-bank-login.com",
            "💰 [Spam] Kiếm tiền nhanh": "Make $5000 working from home with no experience! Reply YES to start earning today.",
            "🏥 [Spam] Ưu đãi thuốc": "Buy cheap medications online without prescription! 80% discount on all products.",
            "🎉 [Spam] Khuyến mãi": "EXCLUSIVE DEAL! Buy 1 get 2 free on all luxury watches this weekend only. Shop now!",
            "📅 [Ham] Lịch hẹn": "Hey, just reminding you about our project alignment meeting tomorrow at 3 PM in conference room B.",
            "👨‍👩‍👧 [Ham] Gia đình": "Mom, I will be home late tonight because of extra work at the office. Don't wait up for dinner.",
            "📚 [Ham] Học tập": "Hi team, please find attached the weekly assignment report and review the feedback before Friday.",
            "✈️ [Ham] Đặt vé": "Your flight VN123 from Hanoi to Ho Chi Minh City has been confirmed. Boarding starts at 08:30 AM.",
            "📦 [Ham] Giao hàng": "Đơn hàng #SP12345 của bạn đã được giao cho đơn vị vận chuyển. Dự kiến giao trong 2 ngày tới."
        }
        
        with st.expander("⚡ Hoặc chọn nhanh các mẫu câu có sẵn tại đây:"):
            selected_sample = st.selectbox("Chọn mẫu câu mẫu:", list(sample_texts.keys()))
            if st.button("Điền nội dung mẫu này"):
                st.session_state["preset_text"] = sample_texts[selected_sample]

        # Khởi tạo giá trị trong session state nếu có
        default_val = st.session_state.get("preset_text", "")
        
        # Ô nhập liệu văn bản
        user_input = st.text_area(
            label="Nội dung tin nhắn / Email",
            value=default_val,
            placeholder="Nhập hoặc dán nội dung vào đây...",
            height=180
        )
        
        predict_clicked = st.button("🚀 Phân tích nội dung", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 Kết quả phân tích & Đánh giá")
        
        if predict_clicked:
            if not user_input.strip():
                st.warning("⚠️ Vui lòng nhập nội dung văn bản để kiểm tra!")
            else:
                with st.spinner("Đang xử lý mô hình..."):
                    # 1. Biến đổi dữ liệu văn bản
                    input_vector = vectorizer.transform([user_input])
                    
                    # 2. Dự đoán nhãn
                    prediction = model.predict(input_vector)[0]
                    
                    # 3. Lấy xác suất dự đoán
                    try:
                        proba = model.predict_proba(input_vector)[0]
                        spam_prob = proba[1]
                        ham_prob = proba[0]
                    except:
                        # Trường hợp mô hình không hỗ trợ predict_proba
                        spam_prob = 1.0 if prediction == 1 else 0.0
                        ham_prob = 1.0 - spam_prob

                # Hiển thị kết quả dạng Card/Metrics
                if prediction == 1:
                    st.error("🚨 **KẾT QUẢ: THƯ RÁC (SPAM)**")
                else:
                    st.success("✅ **KẾT QUẢ: THƯ BÌNH THƯỜNG (HAM)**")
                
                # Hiển thị biểu đồ phân phối xác suất trực quan bằng Progress Bar
                st.markdown("##### 📈 Tỷ lệ phân loại chi tiết:")
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Khả năng là Spam", f"{spam_prob * 100:.2f}%")
                col_m2.metric("Khả năng là Hợp lệ", f"{ham_prob * 100:.2f}%")
                
                st.markdown("Tỷ lệ Spam:")
                st.progress(float(spam_prob))
                
                st.markdown("Tỷ lệ Hợp lệ (Ham):")
                st.progress(float(ham_prob))
        else:
            st.info("👈 Vui lòng nhập nội dung bên trái và bấm **Phân tích nội dung** để xem kết quả đánh giá chi tiết.")