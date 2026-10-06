# --- lib
1. joblib
làm việc với các đối tượng dữ liệu lớn chứa nhiều mảng số (NumPy arrays). Trong lĩnh vực Học máy (Machine Learning), Joblib thường được sử dụng phổ biến để lưu (save) và tải (load) các mô hình đã huấn luyện xuống ổ cứng (thay thế cho hàm pickle truyền thống vì tốc độ xử lý mảng lớn nhanh hơn và hiệu quả hơn).

Dung lượng cài đặt (Package size): Rất nhẹ, chỉ khoảng vài megabytes (MB), do nó chủ yếu tập trung vào việc tối ưu hóa tuần tự hóa dữ liệu.

Dung lượng lưu trữ mô hình (Model file size): Joblib cực kỳ hiệu quả trong việc lưu trữ các mô hình có cấu trúc dữ liệu dạng mảng. Nó hỗ trợ tích hợp sẵn các thuật toán nén dữ liệu (như zlib, lz4, gzip). Khi bạn lưu mô hình với mức nén phù hợp (ví dụ: joblib.dump(model, 'model.joblib', compress=3)), kích thước file lưu trên ổ cứng sẽ giảm đi đáng kể, giúp tiết kiệm không gian lưu trữ và tối ưu tốc độ truyền tải file.


2. xgboost
XGBoost là một thư viện mã nguồn mở tối ưu hóa thuật toán Gradient Boosting dựa trên cấu trúc cây quyết định (decision trees). Đây là một trong những công cụ mạnh mẽ và phổ biến nhất trong các cuộc thi khoa học dữ liệu (như Kaggle) cũng như trong môi trường sản xuất thực tế nhờ tốc độ tính toán siêu nhanh và độ chính xác cao.

Dung lượng cài đặt (Package size): Lớn hơn Joblib khá nhiều (thường dao động từ vài chục đến hơn 100 MB tùy thuộc vào hệ điều hành). Nguyên nhân là vì lõi của XGBoost được viết bằng mã nguồn C++ tối ưu hiệu năng cao, kèm theo các thư viện biên dịch hỗ trợ cả CPU và GPU.

Dung lượng mô hình (Model file size): Kích thước file mô hình XGBoost khi lưu xuống ổ cứng phụ thuộc hoàn toàn vào cấu hình huấn luyện của bạn:

Số lượng cây (n_estimators): Càng nhiều cây, file càng nặng.

Độ sâu cây (max_depth): Cây càng sâu, mô hình càng phức tạp và dung lượng càng lớn.

Thực tế: Các mô hình nhỏ có thể chỉ vài trăm KB, nhưng các mô hình lớn dùng trong bài toán phức tạp (hàng nghìn cây) có thể đạt kích thước từ vài chục MB đến vài trăm MB.

Tối ưu hóa: XGBoost hiện nay hỗ trợ lưu mô hình dưới các định dạng hiện đại như JSON hoặc UBJSON (bên cạnh định dạng nhị phân cũ), giúp tối ưu hóa tốt kích thước file và dễ dàng tương thích đa ngôn ngữ.

# --- stack


# --- more