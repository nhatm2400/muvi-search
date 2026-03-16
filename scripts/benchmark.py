import sys
import os
import time
import glob
import shutil
import argparse

# 1. Thêm root dự án vào hệ thống
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 2. OVERRIDE (ĐÁNH TRÁO) ĐƯỜNG DẪN TRƯỚC KHI IMPORT
# Việc này giúp script không đụng vào thư mục chứa 216 video đang làm việc của bạn
from configs import settings

BENCHMARK_DIR = os.path.join(settings.BASE_DIR, "data", "benchmark_workspace")
settings.KEYFRAME_DIR = os.path.join(BENCHMARK_DIR, "keyframes")
settings.VISUAL_INDEX_PATH = os.path.join(BENCHMARK_DIR, "faiss_siglip.bin")
settings.VISUAL_MAP_PATH = os.path.join(BENCHMARK_DIR, "id_mapping.json")
settings.PROCESSED_LOG_PATH = os.path.join(BENCHMARK_DIR, "processed_videos_log.txt")

# Xóa workspace cũ nếu có để đảm bảo test luôn sạch sẽ
if os.path.exists(BENCHMARK_DIR):
    shutil.rmtree(BENCHMARK_DIR)
os.makedirs(settings.KEYFRAME_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.VISUAL_INDEX_PATH), exist_ok=True)

# 3. IMPORT CÁC MODULE CHÍNH SAU KHI ĐÃ ĐỔI ĐƯỜNG DẪN
from core.visual.indexer import VisualIndexer
from scripts.generate_crops import main as run_yolo_cropping

def count_files(directory, suffix="*.jpg"):
    return len(glob.glob(os.path.join(directory, "**", suffix), recursive=True))

def main():
    parser = argparse.ArgumentParser(description="Benchmark trên 1 video chỉ định")
    parser.add_argument("--video", type=str, required=True, help="Tên file video nằm trong raw_videos (vd: L01_V001.mp4)")
    args = parser.parse_args()

    video_name = args.video
    video_path = os.path.join(settings.VIDEO_DIR, video_name)

    if not os.path.exists(video_path):
        print(f"[LỖI] Không tìm thấy video '{video_name}' tại: {settings.VIDEO_DIR}")
        return

    print(f"=== BẮT ĐẦU BENCHMARK TRÊN VIDEO: {video_name} ===")
    print(f"Workspace test độc lập: {BENCHMARK_DIR}")
    
    indexer = VisualIndexer()
    
    # [1] ĐO THỜI GIAN EXTRACTION: Gọi thẳng hàm xử lý 1 video thay vì chạy vòng lặp
    start_time = time.time()
    print("\n[1] Đang chạy Keyframe Extraction...")
    extracted_count = indexer.extract_keyframes(video_path)
    time_extract = time.time() - start_time
    num_original_frames = count_files(settings.KEYFRAME_DIR)
    
    # [2] ĐO THỜI GIAN CROPPING
    start_time = time.time()
    print("\n[2] Đang chạy YOLOv8 Cropping...")
    run_yolo_cropping()
    time_crop = time.time() - start_time
    
    num_total_frames = count_files(settings.KEYFRAME_DIR)
    num_crops = num_total_frames - num_original_frames
    
    # [3] ĐO THỜI GIAN INDEXING
    start_time = time.time()
    print("\n[3] Đang chạy Vector Embedding & Indexing (SigLIP)...")
    indexer.run_indexing()
    time_index = time.time() - start_time
    
    total_time = time_extract + time_crop + time_index
    
    print("\n==================================================")
    print("             BÁO CÁO BENCHMARK (1 VIDEO)          ")
    print("==================================================")
    print(f"Video test:                   {video_name}")
    print(f"1. Số lượng Keyframe gốc:     {num_original_frames} ảnh")
    print(f"2. Số lượng Crop (YOLO):      {num_crops} ảnh (Tăng {((num_crops)/max(1, num_original_frames))*100:.1f}%)")
    print(f"3. Tổng số vector (FAISS):    {num_total_frames} vector")
    print("--------------------------------------------------")
    print(f"Thời gian Trích xuất cảnh:    {time_extract:.2f} giây")
    print(f"Thời gian Cắt YOLOv8:         {time_crop:.2f} giây")
    print(f"Thời gian Nhúng SigLIP:       {time_index:.2f} giây")
    print(f"TỔNG THỜI GIAN CHẠY MỚI:      {total_time:.2f} giây")
    print(f"TỔNG THỜI GIAN CHẠY CŨ:       {(time_extract + (time_index * (num_original_frames/max(1, num_total_frames)))):.2f} giây (Ước tính)")
    print("==================================================")

if __name__ == "__main__":
    main()