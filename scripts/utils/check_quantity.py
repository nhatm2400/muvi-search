import os
import sys
import glob

# Thêm thư mục gốc vào path để gọi file settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import settings

def main():
    print("=== THỐNG KÊ DỮ LIỆU KEYFRAME & CROP ===")
    
    # Quét toàn bộ ảnh trong thư mục keyframes
    search_pattern = os.path.join(settings.KEYFRAME_DIR, "**", "*.jpg")
    all_images = glob.glob(search_pattern, recursive=True)
    
    if not all_images:
        print("Chưa có ảnh nào trong thư mục data/keyframes!")
        return

    # Khởi tạo từ điển để lưu thống kê theo từng thư mục video
    stats_by_video = {}
    total_originals = 0
    total_crops = 0

    for img_path in all_images:
        # Lấy tên thư mục chứa ảnh (tên video)
        video_name = os.path.basename(os.path.dirname(img_path))
        filename = os.path.basename(img_path)
        
        if video_name not in stats_by_video:
            stats_by_video[video_name] = {"originals": 0, "crops": 0}
            
        # Phân loại ảnh gốc và ảnh crop
        if "_crop" in filename:
            stats_by_video[video_name]["crops"] += 1
            total_crops += 1
        else:
            stats_by_video[video_name]["originals"] += 1
            total_originals += 1

    # In kết quả tổng quan
    print(f"\n[TỔNG QUAN TOÀN HỆ THỐNG]")
    print(f"- Tổng số video đã xử lý : {len(stats_by_video)}")
    print(f"- Tổng số ảnh sinh ra    : {len(all_images)}")
    print(f"- Số ảnh gốc (Frames)    : {total_originals}")
    print(f"- Số ảnh vật thể (Crops) : {total_crops}")
    
    # In chi tiết từng video (Hiển thị dạng bảng đơn giản)
    print("\n[CHI TIẾT TỪNG VIDEO]")
    print(f"{'Tên Video':<20} | {'Ảnh Gốc':<10} | {'Ảnh Crop':<10} | {'Tổng cộng':<10}")
    print("-" * 58)
    
    # Sắp xếp theo tên video cho dễ nhìn
    for video, data in sorted(stats_by_video.items()):
        orig = data["originals"]
        crop = data["crops"]
        total = orig + crop
        print(f"{video:<20} | {orig:<10} | {crop:<10} | {total:<10}")

if __name__ == "__main__":
    main()