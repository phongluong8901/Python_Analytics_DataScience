import streamlit as st
from recommend import df, recommend_songs

# 1. Cấu hình trang (Phải đặt ở dòng đầu tiên của Streamlit)
st.set_page_config(
    page_title="AI Music Recommender 🎧",
    page_icon="🎶",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Tùy chỉnh CSS giao diện chuyên nghiệp (Modern Dark / Clean Theme)
st.markdown("""
    <style>
        /* Tổng thể ứng dụng */
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* Tiêu đề chính */
        .title-container {
            text-align: center;
            padding: 1rem 0 2rem 0;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #8892b0;
            font-size: 1.1rem;
        }

        /* Thẻ hiển thị bài hát đang chọn */
        .selected-song-card {
            background: linear-gradient(135deg, #1f293d 0%, #111827 100%);
            border: 1px solid #374151;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Card kết quả gợi ý */
        .recommendation-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 14px 18px;
            border-radius: 10px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.3s ease;
        }
        .recommendation-card:hover {
            border-color: #1DB954;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(29, 185, 84, 0.15);
        }
        .song-info {
            display: flex;
            flex-direction: column;
        }
        .song-name {
            font-weight: 700;
            font-size: 1.1rem;
            color: #f0f6fc;
        }
        .artist-name {
            color: #8b949e;
            font-size: 0.9rem;
            margin-top: 4px;
        }
        .rank-badge {
            background-color: rgba(29, 185, 84, 0.15);
            color: #1DB954;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            border: 1px solid rgba(29, 185, 84, 0.3);
        }
        
        /* Tùy chỉnh nút bấm */
        div.stButton > button {
            background-color: #1DB954;
            color: white;
            font-weight: 700;
            width: 100%;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            border: none;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #1ed760;
            box-shadow: 0 4px 12px rgba(29, 185, 84, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# 3. Phần Header giao diện
st.markdown("""
    <div class="title-container">
        <div class="main-title">🎧 AI Music Recommender</div>
        <div class="subtitle">Khám phá các bài hát có phong cách lời tương tự dựa trên công nghệ NLP & Machine Learning</div>
    </div>
""", unsafe_allow_html=True)

# 4. Khu vực nhập liệu / lựa chọn bài hát
song_list = sorted(df['song'].dropna().unique())

st.markdown("### 🔍 Chọn bài hát yêu thích của bạn")
selected_song = st.selectbox(
    "Tìm hoặc chọn bài hát:",
    song_list,
    label_visibility="collapsed",
    help="Gõ vài ký tự để tìm kiếm bài hát nhanh hơn"
)

# Lấy thông tin nghệ sĩ tương ứng với bài hát được chọn để hiển thị thêm chi tiết
artist_name = df[df['song'].str.lower() == selected_song.lower()]['artist'].values[0]

st.markdown(f"""
    <div class="selected-song-card">
        <span style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Bài hát đang chọn</span>
        <div style="font-size: 1.3rem; font-weight: bold; color: #ffffff; margin-top: 4px;">🎵 {selected_song}</div>
        <div style="color: #1DB954; font-size: 0.95rem; margin-top: 2px;">🎤 Nghệ sĩ: {artist_name}</div>
    </div>
""", unsafe_allow_html=True)

# 5. Nút bấm kích hoạt gợi ý
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_clicked = st.button("🚀 Gợi ý bài hát tương tự")

# 6. Xử lý kết quả trả về
if search_clicked:
    with st.spinner("🤖 Đang phân tích lời bài hát và tìm kiếm các ca khúc phù hợp..."):
        recommendations = recommend_songs(selected_song, top_n=5)
        
        if recommendations is None:
            st.warning("⚠️ Không tìm thấy bài hát này trong cơ sở dữ liệu.")
        else:
            st.balloons() # Hiệu ứng khi thành công
            st.markdown("### ✨ Top bài hát có phong cách tương tự:")
            
            # Hiển thị kết quả dưới dạng danh sách Card thay vì bảng thông thường
            for idx, row in recommendations.iterrows():
                st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="song-info">
                            <span class="song-name">🎶 {row['song']}</span>
                            <span class="artist-name">👤 {row['artist']}</span>
                        </div>
                        <div class="rank-badge">Top {idx}</div>
                    </div>
                """, unsafe_allow_html=True)

# 7. Chân trang (Footer)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #8b949e; font-size: 0.8rem;'>Powered by Streamlit, TF-IDF, and Cosine Similarity 🚀</p>", 
    unsafe_allow_html=True
)