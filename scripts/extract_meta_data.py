import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import cv2
from tqdm import tqdm
from configs import settings

def main():
    print("EXTRACT VIDEO METADATA")
    
    video_dir = settings.VIDEO_DIR
    metadata_output = os.path.join(settings.INDEX_DIR, "video_metadata.json")
    os.makedirs(settings.INDEX_DIR, exist_ok=True)

    if not os.path.exists(video_dir):
        print(f"Không tìm thấy thư mục video tại: {video_dir}")
        print("Hãy đảm bảo bạn đã copy video vào đúng thư mục 'data/raw_videos'.")
        return

    print("Đang quét thông số video")
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mkv', '.avi', '.mov'))]
    metadata = {}

    if not video_files:
        print(f"Không có video nào định dạng hợp lệ trong {video_dir}")
        return

    for vid in tqdm(video_files, desc="Processing videos"):
        full_path = os.path.join(video_dir, vid)
        cap = cv2.VideoCapture(full_path)
        
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            key_name = os.path.splitext(vid)[0]
            
            metadata[key_name] = {
                "fps": fps,
                "total_frames": total_frames,
                "duration": total_frames / fps if fps > 0 else 0
            }
        else:
            print(f"Không mở được video: {vid}")
        
        cap.release()
    with open(metadata_output, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print(f"\nĐã lưu metadata của {len(metadata)} video vào: '{metadata_output}'")
    
    if metadata:
        print("Ví dụ dữ liệu trích xuất được:")
        print(json.dumps(list(metadata.items())[0], indent=4))

if __name__ == "__main__":
    main()