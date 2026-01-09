# Hướng dẫn chạy chương trình mô phỏng lưới điện

## Yêu cầu môi trường
Đảm bảo đã cài đặt Python và các thư viện cần thiết:
```bash
pip install numpy matplotlib
```

## Các file chương trình
Dự án bao gồm 3 file chính tương ứng với 3 nội dung yêu cầu:

### 1. Thuật toán Khử Gauss (Gaussian Elimination)
File: `gauss_direct.py`
Chức năng: Giải hệ phương trình bằng phương pháp khử Gauss và vẽ biểu đồ kết quả nghiệm.
**Cách chạy:**
```bash
python gauss_direct.py
```

### 2. Thuật toán Gauss-Seidel (Iterative Method)
File: `gauss_seidel.py`
Chức năng: Giải hệ bằng phương pháp lặp và vẽ biểu đồ hội tụ sai số.
**Cách chạy:**
```bash
python gauss_seidel.py
```

### 3. Mô phỏng Lưới điện Tòa nhà (Tổng hợp)
File: `power_grid_simulation.py`
Chức năng: Áp dụng cả 2 thuật toán vào bài toán lưới điện 3 tầng, so sánh kết quả và đưa ra cảnh báo kỹ thuật.
**Cách chạy:**
```bash
python power_grid_simulation.py
```

## Lưu ý
- Khi chạy, các cửa sổ biểu đồ sẽ hiện lên. Bạn cần đóng cửa sổ biểu đồ hiện tại để chương trình có thể kết thúc hoặc chạy tiếp (nếu dùng script gộp).
- Kết quả hình ảnh cũng sẽ được lưu vào thư mục chứa file.
