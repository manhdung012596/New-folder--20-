import numpy as np
import matplotlib.pyplot as plt
import sys

# Thiết lập mã hóa UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def solve_gauss_seidel(A, b, x0, tol=1e-5, max_iter=1000):
    """
    Giải hệ Ax=b bằng phương pháp lặp Gauss-Seidel.
    Trả về: nghiệm x, số bước lặp, lịch sử sai số
    """
    x = x0.copy()
    errors = []
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(len(b)):
            sum_j = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum_j) / A[i, i]
        
        error = np.linalg.norm(x - x_old, ord=np.inf)
        errors.append(error)

        if error < tol:
            return x, k, errors
            
    return x, max_iter, errors

if __name__ == "__main__":
    print("--- B. THUẬT TOÁN GAUSS-SEIDEL (ITERATIVE METHOD) ---")
    
    # Hệ phương trình ví dụ (diag dominant)
    A_demo = np.array([[4.0, -1.0, 0.0],
                       [-1.0, 4.0, -1.0],
                       [0.0, -1.0, 3.0]])
    b_demo = np.array([12.0, -1.0, 10.0])
    
    print(f"Ma trận A demo:\n{A_demo}")
    
    x_gs, steps, error_history = solve_gauss_seidel(A_demo, b_demo, np.zeros(3))
    
    print(f"\nKết quả sau {steps} bước lặp: {x_gs}")
    print(f"Sai số cuối cùng: {error_history[-1]}")

    # Vẽ biểu đồ hội tụ
    plt.figure(figsize=(8, 5))
    plt.plot(error_history, marker='o', markersize=4, color='crimson')
    plt.xlabel('Bước lặp (Iteration)')
    plt.ylabel('Sai số max (Log scale)')
    plt.yscale('log')
    plt.title('Minh họa: Độ hội tụ của Gauss-Seidel')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('gauss_seidel_convergence.png')
    print("Đã lưu biểu đồ vào file 'gauss_seidel_convergence.png'")
    print("Đang hiển thị biểu đồ hội tụ...")
    plt.show()
