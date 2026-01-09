import numpy as np
import matplotlib.pyplot as plt
import sys
# Import thuật toán từ 2 file kia
# Lưu ý: Python cần 2 file kia nằm cùng thư mục hoặc trong path
try:
    from gauss_direct import solve_gauss_direct
    from gauss_seidel import solve_gauss_seidel
except ImportError:
    print("Lỗi: Không tìm thấy file gauss_direct.py hoặc gauss_seidel.py")
    sys.exit(1)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def run_simulation():
    print("--- C. VÍ DỤ ÁP DỤNG: MÔ PHỎNG LƯỚI ĐIỆN TÒA NHÀ ---")

    # Ma trận dẫn nạp A (3 tầng)
    A = np.array([[ 0.5, -0.2,  0.0],
                  [-0.2,  0.8, -0.3],
                  [ 0.0, -0.3,  0.3]])

    # Vector nguồn b (Dòng điện)
    b = np.array([10, 0, 5]) 

    print(f"Ma trận dẫn nạp A:\n{A}")
    print(f"Nguồn dòng kích thích b: {b}")

    # 1. Giải bằng Khử Gauss
    v_direct = solve_gauss_direct(A, b)
    
    # 2. Giải bằng Gauss-Seidel
    v_iter, steps, errors = solve_gauss_seidel(A, b, np.zeros(3))

    # In kết quả
    print("\nKẾT QUẢ SO SÁNH:")
    print(f"Điện áp (Trực tiếp): {v_direct}")
    print(f"Điện áp (Lặp GS):    {v_iter}")
    print(f"Độ sai lệch: {np.linalg.norm(v_direct - v_iter):.6f}")

    node_min = np.argmin(v_direct)
    print(f"CẢNH BÁO: Nút {node_min + 1} có điện áp thấp nhất ({v_direct[node_min]:.2f} V)")

    # 3. Vẽ biểu đồ tổng hợp
    plt.figure(figsize=(10, 6))
    
    # Subplot 1: So sánh điện áp
    plt.subplot(2, 1, 1)
    nodes = np.arange(1, 4)
    plt.bar(nodes - 0.2, v_direct, 0.4, label='Gauss Direct', color='royalblue')
    plt.bar(nodes + 0.2, v_iter, 0.4, label='Gauss-Seidel', color='orange')
    plt.ylabel('Điện áp (V)')
    plt.title('Điện áp tại các nút (Lưới điện tòa nhà)')
    plt.xticks(nodes, [f'Nút {i}' for i in nodes])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Hội tụ của GS
    plt.subplot(2, 1, 2)
    plt.plot(errors, marker='s', markersize=4, color='darkgreen')
    plt.xlabel('Số bước lặp')
    plt.ylabel('Sai số (Log)')
    plt.yscale('log')
    plt.title('Quá trình hội tụ của thuật toán Gauss-Seidel cho lưới điện')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('power_grid_simulation_result.png')
    print("Đã lưu biểu đồ vào file 'power_grid_simulation_result.png'")
    print("Đang hiển thị biểu đồ mô phỏng lưới điện...")
    plt.show()

if __name__ == "__main__":
    run_simulation()
