import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Trỏ vào workspace bạn vừa tạo lúc nãy
from configs import settings
BENCHMARK_DIR = os.path.join(settings.BASE_DIR, "data", "benchmark_workspace")
settings.VISUAL_INDEX_PATH = os.path.join(BENCHMARK_DIR, "faiss_siglip.bin")
settings.VISUAL_MAP_PATH = os.path.join(BENCHMARK_DIR, "id_mapping.json")

from core.visual.searcher import VisualSearcher

def main():
    if not os.path.exists(settings.VISUAL_INDEX_PATH):
        print("[LỖI] Không tìm thấy file Index test.")
        return

    print("=== A/B TESTING: PHƯƠNG PHÁP CŨ vs PHƯƠNG PHÁP MỚI ===")
    searcher = VisualSearcher()
    
    # Các truy vấn tập trung vào vật thể hoặc chi tiết trong video L21_V001
    queries = [
        "một cái bảng hiệu", 
        "chiếc xe máy",
        "khuôn mặt người",
        "người mặc áo đỏ",
        "một dòng chữ trên màn hình"
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"🔍 QUERY: '{q}'")
        print(f"{'='*60}")
        
        # Lấy top 30 để đủ mảng lọc
        results = searcher.search(q, top_k=30)
        
        old_method_results = []
        new_method_results = []
        
        for res in results:
            path = res['path']
            score = res['score']
            
            # Phương pháp cũ: Bỏ qua tất cả các ảnh do YOLO crop
            if "_crop" not in path:
                old_method_results.append(res)
            
            # Phương pháp mới: Lấy tất cả (bao gồm cả crop và ảnh gốc)
            new_method_results.append(res)
            
        print("\n❌ PHƯƠNG PHÁP CŨ (Chỉ dùng ảnh toàn cảnh):")
        if not old_method_results:
            print("   -> Không tìm thấy gì!")
        else:
            for i, res in enumerate(old_method_results[:3]):
                print(f"   Top {i+1} | Điểm: {res['score']:.4f} | File: {os.path.basename(res['path'])}")
                
        print("\n✅ PHƯƠNG PHÁP MỚI (YOLO Crops + Temporal):")
        for i, res in enumerate(new_method_results[:3]):
            # Đánh dấu highlight nếu top 1 là ảnh crop
            marker = "🔥 (YOLO CROP)" if "_crop" in res['path'] else ""
            print(f"   Top {i+1} | Điểm: {res['score']:.4f} | File: {os.path.basename(res['path'])} {marker}")

if __name__ == "__main__":
    main()