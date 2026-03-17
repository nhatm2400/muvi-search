"""
=============================================================================
BENCHMARK TỐC ĐỘ HỆ THỐNG MUVI-SEARCH (VISUAL PIPELINE)
=============================================================================
File này gộp cả 2 tác vụ: Đo tốc độ Indexing (Tạo dữ liệu) và Searching (Tìm kiếm).
Sử dụng một workspace hoàn toàn độc lập để không ảnh hưởng đến data thật.

Cách chạy:
1. Chạy cả Indexing và Search trên 1 video:
   python scripts/evaluation/bench_speed.py --mode all --video L01_V001.mp4

2. Chỉ đo tốc độ Indexing:
   python scripts/evaluation/bench_speed.py --mode index --video L01_V001.mp4

3. Chỉ đo tốc độ Tìm kiếm (Yêu cầu đã chạy lệnh Indexing trước đó):
   python scripts/evaluation/bench_speed.py --mode search
=============================================================================
"""

import sys
import os
import time
import glob
import shutil
import argparse

# 1. Thêm root dự án vào hệ thống (Lên 2 cấp từ scripts/evaluation)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# 2. OVERRIDE ĐƯỜNG DẪN TRƯỚC KHI IMPORT CÁC MODULE LÕI
# Đảm bảo benchmark chạy trên workspace an toàn
from configs import settings

BENCHMARK_DIR = os.path.join(settings.BASE_DIR, "data", "benchmark_workspace")
settings.KEYFRAME_DIR = os.path.join(BENCHMARK_DIR, "keyframes")
settings.VISUAL_INDEX_PATH = os.path.join(BENCHMARK_DIR, "faiss_siglip.bin")
settings.VISUAL_MAP_PATH = os.path.join(BENCHMARK_DIR, "id_mapping.json")
settings.PROCESSED_LOG_PATH = os.path.join(BENCHMARK_DIR, "processed_videos_log.txt")

# 3. IMPORT CÁC MODULE CHÍNH (Sau khi đã override settings)
from core.visual.indexer import VisualIndexer
from core.visual.searcher import VisualSearcher
from core.visual.cropper import main as run_yolo_cropping # Đã cập nhật đường dẫn mới

def count_files(directory, suffix="*.jpg"):
    return len(glob.glob(os.path.join(directory, "**", suffix), recursive=True))

def run_indexing_benchmark(video_name):
    print(f"\n=== BẮT ĐẦU BENCHMARK INDEXING TRÊN VIDEO: {video_name} ===")
    
    video_path = os.path.join(settings.VIDEO_DIR, video_name)
    if not os.path.exists(video_path):
        print(f"[LỖI] Không tìm thấy video '{video_name}' tại: {settings.VIDEO_DIR}")
        return False

    # Xóa workspace cũ và tạo lại sạch sẽ cho phần Indexing
    if os.path.exists(BENCHMARK_DIR):
        shutil.rmtree(BENCHMARK_DIR)
    os.makedirs(settings.KEYFRAME_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(settings.VISUAL_INDEX_PATH), exist_ok=True)

    indexer = VisualIndexer()
    
    # [1] EXTRACTION
    print("\n[1] Đang chạy Keyframe Extraction...")
    start_time = time.time()
    indexer.extract_keyframes(video_path)
    time_extract = time.time() - start_time
    num_original_frames = count_files(settings.KEYFRAME_DIR)
    
    # [2] CROPPING
    print("\n[2] Đang chạy YOLOv8 Cropping...")
    start_time = time.time()
    run_yolo_cropping()
    time_crop = time.time() - start_time
    
    num_total_frames = count_files(settings.KEYFRAME_DIR)
    num_crops = num_total_frames - num_original_frames
    
    # [3] INDEXING
    print("\n[3] Đang chạy Vector Embedding & Indexing (SigLIP)...")
    start_time = time.time()
    indexer.run_indexing()
    time_index = time.time() - start_time
    
    total_time = time_extract + time_crop + time_index
    
    print("\n==================================================")
    print("             BÁO CÁO INDEXING (1 VIDEO)           ")
    print("==================================================")
    print(f"Video test:                   {video_name}")
    print(f"1. Số lượng Keyframe gốc:     {num_original_frames} ảnh")
    print(f"2. Số lượng Crop (YOLO):      {num_crops} ảnh")
    print(f"3. Tổng số vector (FAISS):    {num_total_frames} vector")
    print("--------------------------------------------------")
    print(f"Thời gian Trích xuất cảnh:    {time_extract:.2f} giây")
    print(f"Thời gian Cắt YOLOv8:         {time_crop:.2f} giây")
    print(f"Thời gian Nhúng SigLIP:       {time_index:.2f} giây")
    print(f"TỔNG THỜI GIAN:               {total_time:.2f} giây")
    print("==================================================")
    return True

def run_search_benchmark():
    print("\n=== BENCHMARK TỐC ĐỘ TRUY XUẤT (RETRIEVAL) ===")
    
    if not os.path.exists(settings.VISUAL_INDEX_PATH):
        print(f"[LỖI] Không tìm thấy file Index tại: {settings.VISUAL_INDEX_PATH}")
        print("Hãy chạy lệnh với mode 'index' hoặc 'all' trước để tạo data test.")
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
    
    # Warm-up GPU (Chạy mồi để GPU sẵn sàng)
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

def main():
    parser = argparse.ArgumentParser(description="Đo lường tốc độ Muvi-Search Pipeline")
    parser.add_argument("--mode", type=str, choices=['index', 'search', 'all'], required=True, 
                        help="Chọn chế độ: index (Tạo dữ liệu), search (Tìm kiếm), all (Cả hai)")
    parser.add_argument("--video", type=str, 
                        help="Tên file video nằm trong raw_videos (Yêu cầu nếu mode là 'index' hoặc 'all')")
    
    args = parser.parse_args()

    if args.mode in ['index', 'all']:
        if not args.video:
            print("[LỖI] Phải cung cấp tên video (--video) khi chạy chế độ 'index' hoặc 'all'")
            return
        success = run_indexing_benchmark(args.video)
        if not success:
            return

    if args.mode in ['search', 'all']:
        run_search_benchmark()

if __name__ == "__main__":
    main()