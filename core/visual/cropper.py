import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from configs import settings

def main():
    print("REGION CROPPING BẰNG YOLO")    
    print("Đang tải model YOLOv8n...")
    model = YOLO('yolov8n.pt') 
    search_pattern = os.path.join(settings.KEYFRAME_DIR, "**", "*.jpg")
    all_images = glob.glob(search_pattern, recursive=True)    
    original_images = [img for img in all_images if "_crop" not in img]
    print(f"Tìm thấy {len(original_images)} keyframes gốc. Bắt đầu phát hiện vật thể...")

    total_crops = 0

    for img_path in tqdm(original_images, desc="Cropping regions"):
        img = cv2.imread(img_path)
        if img is None: continue
        results = model.predict(img, conf=0.6, verbose=False)
        
        crop_idx = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) < 120 or (y2 - y1) < 120:
                    continue
                crop_img = img[y1:y2, x1:x2]
                base_name, ext = os.path.splitext(img_path)
                crop_path = f"{base_name}_crop{crop_idx}{ext}"
                cv2.imwrite(crop_path, crop_img)
                crop_idx += 1
                total_crops += 1

    print(f"\nĐã tạo thêm {total_crops} ảnh crops chi tiết từ keyframes gốc.")

if __name__ == "__main__":
    main()