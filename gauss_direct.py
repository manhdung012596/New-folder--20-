import numpy as np
import matplotlib.pyplot as plt
import sys

# Thiết lập mã hóa UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def solve_gauss_direct(A, b):
    """
    Giải hệ phương trình Ax=b bằng phương pháp khử Gauss.
    """
    n = len(b)
    # Ghép ma trận tăng cường
    Ab = np.column_stack((A, b.astype(float)))

    # Khử xuôi
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Thế ngược
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

if __name__ == "__main__":
    print("--- A. THUẬT TOÁN KHỬ GAUSS (GAUSSIAN ELIMINATION) ---")
    
    # Ví dụ ngẫu nhiên để minh họa thuật toán
    np.random.seed(42)
    A_demo = np.random.rand(5, 5) + np.eye(5) * 2 # Đảm bảo đường chéo trội
    b_demo = np.random.rand(5) * 10
    
    print(f"Ma trận A demo (5x5):\n{A_demo}")
    print(f"Vector b demo:\n{b_demo}")
    
    x_result = solve_gauss_direct(A_demo, b_demo)
    print(f"\nKết quả x: {x_result}")

    # Vẽ biểu đồ kết quả
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 6), x_result, color='teal')
    plt.xlabel('Chỉ số biến (x_i)')
    plt.ylabel('Giá trị')
    plt.title('Minh họa kết quả: Thuật toán Khử Gauss')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('gauss_direct_result.png')
    print("Đã lưu biểu đồ vào file 'gauss_direct_result.png'")
    print("Đang hiển thị biểu đồ minh họa...")
    plt.show()
