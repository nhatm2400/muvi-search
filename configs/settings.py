import os
import torch

# 1. ĐỊNH NGHĨA CÁC THƯ MỤC GỐC
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw_videos", "video") 
KEYFRAME_DIR = os.path.join(BASE_DIR, "data", "keyframes")
INDEX_DIR = os.path.join(BASE_DIR, "data", "indices")  # Đã đổi thành INDEX_DIR cho chuẩn

# Định nghĩa các thư mục con trong indices
VISUAL_DIR = os.path.join(INDEX_DIR, "visual")
OCR_DIR = os.path.join(INDEX_DIR, "ocr")

# 2. ĐƯỜNG DẪN CÁC FILE INDEX & LOG
VISUAL_INDEX_PATH = os.path.join(VISUAL_DIR, "faiss_siglip.bin")
VISUAL_MAP_PATH = os.path.join(VISUAL_DIR, "id_mapping.json")
PROCESSED_LOG_PATH = os.path.join(INDEX_DIR, "processed_videos_log.txt")

# Khai báo đường dẫn OCR đã sửa lỗi tên biến
OCR_INDEX_PATH = os.path.join(OCR_DIR, "sota_index.bin")

# 3. CẤU HÌNH MODEL & THÔNG SỐ
MODEL_NAME = 'ViT-B-16-SigLIP'
PRETRAINED_WEIGHTS = 'webli'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_WORKERS = 0

# 4. TẠO THƯ MỤC TỰ ĐỘNG (Bổ sung thêm thư mục OCR)
os.makedirs(KEYFRAME_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)