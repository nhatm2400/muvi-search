import sys
import os
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# --- THÊM ĐOẠN NÀY ĐỂ TRỎ VÀO WORKSPACE BẠN VỪA TẠO ---
from configs import settings
BENCHMARK_DIR = os.path.join(settings.BASE_DIR, "data", "benchmark_workspace")
settings.VISUAL_INDEX_PATH = os.path.join(BENCHMARK_DIR, "faiss_siglip.bin")
settings.VISUAL_MAP_PATH = os.path.join(BENCHMARK_DIR, "id_mapping.json")
# --------------------------------------------------------

from core.visual.searcher import VisualSearcher

def main():
    print("=== BENCHMARK TỐC ĐỘ TRUY XUẤT (RETRIEVAL) ===")
    
    if not os.path.exists(settings.VISUAL_INDEX_PATH):
        print(f"[LỖI] Không tìm thấy file Index tại: {settings.VISUAL_INDEX_PATH}")
        print("Hãy đảm bảo bạn đã chạy file benchmark.py trước để tạo data test.")
        return

    print("Đang tải SigLIP model và FAISS index vào RAM...")
    start_load = time.time()
    searcher = VisualSearcher()
    print(f"-> Tải xong hệ thống trong {time.time() - start_load:.2f} giây.\n")
    
    queries = [
        "một người đàn ông mặc áo trắng",
        "chiếc xe ô tô đang chạy trên đường",
        "bảng hiệu có chữ",
        "hai người đang nói chuyện",
        "cảnh thành phố về đêm"
    ]
    
    print("--- BẮT ĐẦU ĐO TỐC ĐỘ THỰC TẾ ---")
    total_time = 0
    
    # Warm-up GPU
    _ = searcher.search("hello", top_k=1)
    
    for i, q in enumerate(queries):
        start_q = time.time()
        results = searcher.search(q, top_k=5)
        t = time.time() - start_q
        total_time += t
        print(f"[{i+1}] Query: '{q}'")
        print(f"    -> Phản hồi: {t:.4f} giây (Tìm thấy {len(results)} kết quả)")
        
    print("\n==================================================")
    print(f"TỔNG KẾT: TRUNG BÌNH MỘT TRUY VẤN TỐN {total_time/len(queries):.4f} GIÂY")
    print("==================================================")

if __name__ == "__main__":
    main()