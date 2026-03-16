import os
import json
import random
import cv2

# Đường dẫn tới thư mục ảnh và file JSON
KEYFRAME_DIR = os.path.join("data", "keyframes")
GT_PATH = os.path.join("data", "ground_truth.json")

def main():
    print("=== TOOL TẠO GROUND TRUTH NHANH ===")
    print("Hướng dẫn: Gõ mô tả cho ảnh hiện lên. Bỏ trống và bấm Enter để bỏ qua ảnh. Gõ 'exit' để thoát và lưu.")
    
    # Load data cũ nếu có (Fix luôn lỗi file rỗng)
    ground_truth = {}
    if os.path.exists(GT_PATH):
        try:
            with open(GT_PATH, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
        except json.JSONDecodeError:
            ground_truth = {}

    # Lấy danh sách toàn bộ ảnh
    all_images = []
    for root, dirs, files in os.walk(KEYFRAME_DIR):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                all_images.append(os.path.join(root, file))

    random.shuffle(all_images)
    count = 0

    for img_path in all_images:
        filename = os.path.basename(img_path)
        
        # Bỏ qua nếu ảnh này đã có trong ground truth
        if any(filename in imgs for imgs in ground_truth.values()):
            continue

        # Đọc và resize ảnh
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (int(w * 500 / h), 500))
        
        # Hiển thị ảnh
        cv2.imshow("Tao Ground Truth (Xem anh va go vao Terminal)", img_resized)
        
        # FIX LỖI CRASH CỬA SỔ: Cho OpenCV 10 mili-giây để kịp vẽ ảnh
        cv2.waitKey(10)

        # Chờ nhập câu hỏi từ terminal
        query = input(f"[{count+1}] Nhập mô tả cho '{filename}': ").strip()
        
        if query.lower() == 'exit':
            break
        elif query != "":
            # Nếu người dùng nhập mô tả, lưu lại
            ground_truth[query] = [filename]
            count += 1
            print(f" -> Đã lưu: '{query}'")

    cv2.destroyAllWindows()

    # Lưu file JSON
    os.makedirs(os.path.dirname(GT_PATH), exist_ok=True)
    with open(GT_PATH, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=4, ensure_ascii=False)
    
    print(f"\n[THÀNH CÔNG] Đã lưu {len(ground_truth)} câu truy vấn vào {GT_PATH}")

if __name__ == "__main__":
    main()