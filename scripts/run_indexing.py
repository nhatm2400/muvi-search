import sys
import os
import argparse

# Thêm thư mục gốc vào hệ thống
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual.indexer import VisualIndexer
# Import hàm chạy YOLO từ file generate_crops
from scripts.generate_crops import main as run_yolo_cropping

def main():
    # Cấu hình ArgumentParser để nhận lệnh từ terminal
    parser = argparse.ArgumentParser(description="isual Indexing Pipeline")
    parser.add_argument(
        '--step', 
        type=str, 
        choices=['extract', 'crop', 'index', 'all'], 
        default='all',
        help="Chọn bước để chạy: 'extract', 'crop', 'index', hoặc 'all' (mặc định)"
    )
    args = parser.parse_args()

    print(f"VISUAL PIPELINE (Chế độ: {args.step.upper()})")
    
    indexer = VisualIndexer()
    
    # BƯỚC 1: Trích xuất khung hình
    if args.step in ['extract', 'all']:
        print("\nBƯỚC 1: KEYFRAME EXTRACTION (TEMPORAL) ---")
        indexer.run_extraction()    
    
    # BƯỚC 2: Cắt vật thể nhỏ bằng YOLO
    if args.step in ['crop', 'all']:
        print("\nBƯỚC 2: REGION CROPPING (YOLOv8) ---")
        run_yolo_cropping()

    # BƯỚC 3: Nhúng vector
    if args.step in ['index', 'all']:
        print("\nBƯỚC 3: VECTOR EMBEDDING & FAISS INDEXING ---")
        indexer.run_indexing()
    
    print("\nCOMPLETED")

if __name__ == "__main__":
    main()