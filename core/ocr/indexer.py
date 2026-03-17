import os
import glob
import pickle
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from configs import settings

class OCRIndexer:
    def __init__(self):
        print("--- Khởi tạo OCR Indexer (Qwen2-VL) ---")
        self.device = settings.DEVICE if hasattr(settings, 'DEVICE') else ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def build_index(self):
        all_results = []
        if os.path.exists(settings.OCR_INDEX_PATH):
            with open(settings.OCR_INDEX_PATH, "rb") as f:
                all_results = pickle.load(f)
            print(f"Đã load {len(all_results)} kết quả OCR cũ từ {settings.OCR_INDEX_PATH}.")
            
        indexed_frames = {res['frame_id'] for res in all_results}
        video_dirs = [d for d in os.listdir(settings.KEYFRAME_DIR) if os.path.isdir(os.path.join(settings.KEYFRAME_DIR, d))]
        
        if not video_dirs:
            print(f"Cảnh báo: Không tìm thấy thư mục ảnh nào trong {settings.KEYFRAME_DIR}.")
            return
            
        print(f"Bắt đầu OCR cho {len(video_dirs)} thư mục video...")
        
        for vid_id in video_dirs:
            vid_img_dir = os.path.join(settings.KEYFRAME_DIR, vid_id)
            img_files = glob.glob(os.path.join(vid_img_dir, "*.jpg"))
            
            print(f"Đang xử lý {len(img_files)} ảnh của video: {vid_id}")
            for img_path in tqdm(img_files, desc=f"OCR {vid_id}"):
                frame_id = os.path.basename(img_path)
                
                if frame_id in indexed_frames:
                    continue

                try:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": img_path}, 
                        {"type": "text", "text": "Extract all visible text."}
                    ]}]
                    
                    text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, _ = process_vision_info(messages)
                    inputs = self.processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True)
                        embeds = torch.nn.functional.normalize(outputs.hidden_states[-1].squeeze(0), p=2, dim=-1)
                        
                    all_results.append({
                        "frame_id": frame_id,
                        "embeddings": embeds.to(torch.float16).cpu().numpy()
                    })
                except Exception as e:
                    print(f"Lỗi tại {img_path}: {e}")

        os.makedirs(os.path.dirname(settings.OCR_INDEX_PATH), exist_ok=True)
        
        with open(settings.OCR_INDEX_PATH, "wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"HOÀN THÀNH OCR Indexing! Tổng cộng: {len(all_results)} vector embeddings.")