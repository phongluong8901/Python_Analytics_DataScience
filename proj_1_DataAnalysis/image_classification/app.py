import os
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import pickle

# Thiết lập tiêu đề trang web
st.set_page_config(
    page_title="Crop Classifier App",
    page_icon="🌱",
    layout="centered"
)

# ⚠️ Đổi sang tên file model nâng cấp mới của bạn
MODEL_PATH = "Crop_classifier_upgraded_model.pkl"

@st.cache_resource
def load_model_and_classes():
    """Load model nâng cấp an toàn bằng cách chặn và ép map_location về CPU ngay lúc unpickle"""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    
    # 1. Xác định thiết bị hiện tại ở local
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # 2. Tạo custom unpickler để can thiệp việc load tensor từ file joblib/pickle
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=False)
            return super().find_class(module, name)

    # 3. Đọc file checkpoint
    try:
        with open(MODEL_PATH, "rb") as f:
            checkpoint = CPU_Unpickler(f).load()
    except Exception as e:
        import joblib
        checkpoint = joblib.load(MODEL_PATH)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # 4. Khởi tạo kiến trúc ResNet18 (Phải khớp với lúc train bản nâng cấp)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 5. Load state_dict vào model
    state_dict = checkpoint["model_state_dict"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new_state_dict[k] = v.to(device)
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval() # Chế độ dự đoán (Inference)
    
    return model, idx_to_class, device

# Giao diện Streamlit
st.title("🌱 Phân Loại Cây Trồng Bằng AI (Bản Nâng Cấp)")
st.write("Tải lên hình ảnh lá, thân hoặc quả của cây trồng để hệ thống nhận diện thuộc 140 loại phổ biến.")

model, idx_to_class, device = load_model_and_classes()

if model is None:
    st.error(f"⚠️ Không tìm thấy file model `{MODEL_PATH}`! Hãy chắc chắn bạn đã đặt file checkpoint mới nhất cùng thư mục với `app.py`.")
else:
    # Khu vực upload ảnh
    uploaded_file = st.file_uploader("Chọn một hình ảnh cây trồng (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
            
        with col2:
            st.write("⏳ Đang phân tích...")
            
            # Transform đồng bộ với val_test_transform lúc train
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Dự đoán
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                conf, predicted_idx = torch.max(probabilities, 1)
                
            predicted_class = idx_to_class[predicted_idx.item()]
            confidence = conf.item() * 100
            
            st.success("✨ Kết quả dự đoán:")
            st.markdown(f"### **Tên cây trồng:** `{predicted_class}`")
            st.markdown(f"**Độ tin cậy:** `{confidence:.2f}%`")
            
            # Hiển thị top 3 dự đoán
            st.write("---")
            st.write("📊 **Top dự đoán khả dĩ:**")
            topk_conf, topk_indices = torch.topk(probabilities, 3)
            for i in range(3):
                cls_name = idx_to_class[topk_indices[0][i].item()]
                cls_conf = topk_conf[0][i].item() * 100
                st.write(f"- {cls_name}: **{cls_conf:.2f}%**")