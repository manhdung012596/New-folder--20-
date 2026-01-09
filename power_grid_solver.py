import numpy as np
import sys
import matplotlib.pyplot as plt

# Thiết lập mã hóa UTF-8 cho xuất dữ liệu trên Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def solve_gauss_direct(A, b):
    """
    Giải hệ phương trình lưới điện bằng phương pháp khử Gauss.
    A: Ma trận dẫn nạp (nxn)
    b: Vector nguồn dòng/áp (n)
    """
    n = len(b)
    # Ghép ma trận tăng cường [A|b]
    Ab = np.column_stack((A, b.astype(float)))

    # Quá trình khử xuôi
    for i in range(n):
        # Chọn phần tử trục (pivoting) để tăng độ ổn định số
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Quá trình thế ngược
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

def solve_gauss_seidel(A, b, x0, tol=1e-5, max_iter=1000):
    """
    Giải hệ phương trình lưới điện bằng phương pháp lặp Gauss-Seidel.
    Trả về: nghiệm x, số bước lặp, và lịch sử sai số
    """
    x = x0.copy()
    errors = []
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(len(b)):
            # Công thức lặp GS
            sum_j = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum_j) / A[i, i]
        
        # Tính sai số hiện tại
        error = np.linalg.norm(x - x_old, ord=np.inf)
        errors.append(error)

        # Kiểm tra điều kiện hội tụ
        if error < tol:
            return x, k, errors
            
    return x, max_iter, errors

def visualize_results(v_direct, v_iter, errors):
    """
    Vẽ biểu đồ so sánh kết quả và độ hội tụ.
    """
    plt.figure(figsize=(12, 5))

    # 1. Biểu đồ cột so sánh điện áp
    plt.subplot(1, 2, 1)
    nodes = np.arange(1, len(v_direct) + 1)
    width = 0.35
    
    plt.bar(nodes - width/2, v_direct, width, label='Gauss Direct', color='royalblue')
    plt.bar(nodes + width/2, v_iter, width, label='Gauss-Seidel', color='orange')
    
    plt.xlabel('Nút (Node)')
    plt.ylabel('Điện áp (V)')
    plt.title('So sánh Điện áp tại các Nút')
    plt.xticks(nodes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Biểu đồ hội tụ sai số
    plt.subplot(1, 2, 2)
    plt.plot(errors, marker='o', markersize=4, color='crimson')
    plt.xlabel('Bước lặp (Iteration)')
    plt.ylabel('Sai số max (Log scale)')
    plt.yscale('log')
    plt.title('Độ hội tụ của Gauss-Seidel')
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ma trận dẫn nạp A (đơn vị: Siemens)
    # Đường chéo chính: Tổng điện dẫn tại nút đó
    # Các phần tử khác: Điện dẫn giữa các nút (giá trị âm)
    A = np.array([[ 0.5, -0.2,  0.0],
                  [-0.2,  0.8, -0.3],
                  [ 0.0, -0.3,  0.3]])

    # Vector b: Nguồn dòng kích thích (Ampe) tại các tầng
    b = np.array([10, 0, 5]) 

    print("--- Bắt đầu giải hệ phương trình lưới điện ---")
    print(f"Ma trận dẫn nạp A:\n{A}")
    print(f"Vector nguồn b: {b}")

    # 1. Giải bằng phương pháp trực tiếp
    v_direct = solve_gauss_direct(A, b)
    
    # 2. Giải bằng phương pháp lặp
    v_iter, steps, errors = solve_gauss_seidel(A, b, np.zeros(3))

    print("\n--- KẾT QUẢ ---")
    print(f"1. Phương pháp Trực tiếp (Gauss Elimination): {v_direct}")
    print(f"2. Phương pháp Lặp (Gauss-Seidel): {v_iter} (Hội tụ sau {steps} bước)")

    # D. Kiểm soát và Hậu xử lý (Post-processing)
    print("\n--- HẬU XỬ LÝ & KIỂM TRA ---")
    
    # Kiểm tra tính bảo toàn: Sai số tổng dòng điện tại các nút (A*x - b)
    residual_direct = np.dot(A, v_direct) - b
    residual_iter = np.dot(A, v_iter) - b
    print(f"Sai số dư (Residual) - Trực tiếp: {np.linalg.norm(residual_direct)}")
    print(f"Sai số dư (Residual) - Lặp GS: {np.linalg.norm(residual_iter)}")

    # Đánh giá độ hội tụ: So sánh kết quả giữa hai phương pháp
    diff_percent = np.linalg.norm(v_direct - v_iter) / np.linalg.norm(v_direct) * 100
    print(f"Độ lệch giữa hai phương pháp: {diff_percent:.6f}%")
    if diff_percent < 3:
        print("-> ĐÁNH GIÁ: Kết quả khớp nhau tốt (< 3%)")
    else:
        print("-> CẢNH BÁO: Sai số lớn giữa hai phương pháp!")

    # Trích xuất chỉ số: Xác định nút có điện áp thấp nhất
    min_v_idx = np.argmin(v_direct)
    print(f"Nút có điện áp thấp nhất: Nút {min_v_idx + 1} (Điện áp: {v_direct[min_v_idx]:.4f} V)")
    print("-> Đây là nút yếu nhất trong tòa nhà, cần lưu ý.")

    # Vẽ biểu đồ
    print("\nĐang hiển thị biểu đồ...")
    visualize_results(v_direct, v_iter, errors)
