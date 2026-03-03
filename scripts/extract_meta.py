import os
import json
import cv2
from tqdm import tqdm

VIDEO_DIR = r"E:\FPTU\Spring26\DAT301m - SLP301\Dataset_Visual_Extraction\Data\Video\video"
METADATA_OUTPUT = "dict/video_metadata.json"

if not os.path.exists("dict"):
    os.makedirs("dict")

print("Đang quét thông số video (FPS)...")

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mkv', '.avi', '.mov'))]
metadata = {}

for vid in tqdm(video_files):
    full_path = os.path.join(VIDEO_DIR, vid)
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
        print(f"⚠️ Không mở được video: {vid}")
    
    cap.release()

with open(METADATA_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)

print(f"\nĐã lưu metadata của {len(metadata)} video vào '{METADATA_OUTPUT}'")
print("Ví dụ dữ liệu:", json.dumps(list(metadata.items())[0], indent=4))