import json
import streamlit as st
from recommend import df, recommend_movies
from omdb_utils import get_movie_details

# 1. Cấu hình trang (Phải đặt ở dòng đầu tiên)
st.set_page_config(
    page_title="Cinema AI - Movie Recommender",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Tùy chỉnh CSS giao diện chuyên nghiệp (Cinema Dark Theme)
st.markdown("""
    <style>
        /* Tổng thể ứng dụng */
        .main {
            background-color: #0b0f19;
            color: #f3f4f6;
        }
        
        /* Tiêu đề ứng dụng */
        .title-container {
            text-align: center;
            padding: 1.5rem 0 2rem 0;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #e50914 0%, #ff4b4b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
            letter-spacing: -0.5px;
        }
        .subtitle {
            color: #9ca3af;
            font-size: 1.05rem;
        }

        /* Thẻ Card bao bọc kết quả phim */
        .movie-card {
            background-color: #131b2e;
            border: 1px solid #1f293d;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .movie-card:hover {
            border-color: #e50914;
            transform: translateY(-3px);
            box-shadow: 0 10px 20px -5px rgba(229, 9, 20, 0.2);
        }

        /* Huy hiệu xếp hạng */
        .rank-badge {
            display: inline-block;
            background-color: rgba(229, 9, 20, 0.15);
            color: #ff4b4b;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.8rem;
            border: 1px solid rgba(229, 9, 20, 0.3);
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }

        /* Typography cho phim */
        .movie-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 6px;
        }
        .movie-plot {
            color: #9ca3af;
            font-size: 0.92rem;
            line-height: 1.5;
        }

        /* Tùy chỉnh nút bấm chính */
        div.stButton > button {
            background-color: #e50914;
            color: white;
            font-weight: 700;
            width: 100%;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(229, 9, 20, 0.3);
        }
        div.stButton > button:hover {
            background-color: #f6121d;
            box-shadow: 0 6px 16px rgba(229, 9, 20, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

# Đọc cấu hình API
try:
    config = json.load(open("config.json"))
    OMDB_API_KEY = config["OMDB_API_KEY"]
except Exception:
    OMDB_API_KEY = None

# 3. Phần Header giao diện
st.markdown("""
    <div class="title-container">
        <div class="main-title">🎬 Cinema AI Recommender</div>
        <div class="subtitle">Khám phá các bộ phim có cốt truyện và phong cách tương tự qua hệ thống Machine Learning</div>
    </div>
""", unsafe_allow_html=True)

# 4. Khu vực chọn phim
try:
    movie_list = sorted(df['title'].dropna().unique())
except Exception:
    movie_list = []

st.markdown("### 🔍 Chọn bộ phim bạn yêu thích")
selected_movie = st.selectbox(
    "Tìm hoặc chọn phim:",
    movie_list,
    label_visibility="collapsed",
    help="Gõ tên phim để hệ thống tìm kiếm nhanh"
)

# 5. Nút kích hoạt tìm kiếm
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_clicked = st.button("🚀 Gợi ý phim tương tự")

# 6. Xử lý kết quả và hiển thị giao diện Card chuyên nghiệp
if search_clicked:
    with st.spinner("🍿 Đang phân tích dữ liệu và tìm kiếm các tác phẩm điện ảnh phù hợp..."):
        recommendations = recommend_movies(selected_movie)
        
        if recommendations is None or recommendations.empty:
            st.warning("⚠️ Không tìm thấy gợi ý nào phù hợp.")
        else:
            st.balloons() # Hiệu ứng chúc mừng khi load xong
            st.markdown("### ✨ Top bộ phim đề xuất dành riêng cho bạn:")
            
            # Duyệt qua danh sách phim gợi ý và đưa vào từng Card giao diện riêng biệt
            for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
                movie_title = row['title']
                
                # Lấy thông tin chi tiết qua OMDB API (nếu có key)
                plot, poster = "N/A", "N/A"
                if OMDB_API_KEY:
                    plot, poster = get_movie_details(movie_title, OMDB_API_KEY)

                # Ảnh mặc định nếu không có poster hoặc lỗi kết nối
                default_poster = "https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=300&auto=format&fit=crop"
                final_poster = poster if poster and poster != "N/A" else default_poster

                # Tạo khối hiển thị dạng Card cách điệu
                with st.container():
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    
                    c1, c2 = st.columns([1, 3.5])
                    with c1:
                        # Hiển thị poster phim với bo góc tinh tế
                        st.image(final_poster, use_container_width=True)
                    with c2:
                        st.markdown(f'<div class="rank-badge">TOP {rank} ĐỀ XUẤT</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="movie-title">{movie_title}</div>', unsafe_allow_html=True)
                        
                        plot_text = plot if plot != "N/A" else "Chưa có thông tin tóm tắt cốt truyện cho bộ phim này."
                        st.markdown(f'<div class="movie-plot">{plot_text}</div>', unsafe_allow_html=True)
                        
                    st.markdown('</div>', unsafe_allow_html=True)

# 7. Chân trang (Footer)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6b7280; font-size: 0.8rem;'>Powered by Streamlit, OMDB API, and Cosine Similarity 🎥</p>", 
    unsafe_allow_html=True
)