import os
import glob
import pickle
import torch
import whisper
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from configs import settings

class ASRIndexer:
    def __init__(self):
        print("--- Khởi tạo ASR Indexer (Whisper + PhoBERT) ---")
        self.device = settings.DEVICE if hasattr(settings, 'DEVICE') else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Đang tải mô hình Whisper (medium)...")
        self.asr_model = whisper.load_model("medium").to(self.device)
        
        print("Đang tải mô hình PhoBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)

    def build_index(self):
        asr_index = []
        if os.path.exists(settings.ASR_INDEX_PATH):
            with open(settings.ASR_INDEX_PATH, "rb") as f:
                asr_index = pickle.load(f)
            print(f"Đã load {len(asr_index)} đoạn hội thoại cũ từ {settings.ASR_INDEX_PATH}.")

        indexed_videos = {res['video_id'] for res in asr_index}
        video_files = glob.glob(os.path.join(settings.VIDEO_DIR, "*.mp4"))
        
        if not video_files:
            print(f"Cảnh báo: Không tìm thấy video nào trong thư mục {settings.VIDEO_DIR} để xử lý.")
            return

        print(f"Bắt đầu xử lý ASR cho {len(video_files)} video...")
        
        for vid_path in video_files:
            vid_name = os.path.basename(vid_path)
            
            if vid_name in indexed_videos:
                print(f"Video {vid_name} đã được index ASR trước đó, bỏ qua.")
                continue

            print(f"Đang trích xuất giọng nói (Whisper): {vid_name}")
            result = self.asr_model.transcribe(vid_path, language="vi")
            for segment in tqdm(result['segments'], desc=f"PhoBERT Embedding {vid_name}"):
                text = segment['text']
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                with torch.no_grad():
                    out = self.phobert(**inputs)
                    vec = out.last_hidden_state[:, 0, :].cpu().numpy()
                
                asr_index.append({
                    "video_id": vid_name,
                    "start_time": segment['start'],
                    "text": text,
                    "embedding": vec
                })

        os.makedirs(os.path.dirname(settings.ASR_INDEX_PATH), exist_ok=True)
        with open(settings.ASR_INDEX_PATH, "wb") as f:
            pickle.dump(asr_index, f)
            
        print(f"HOÀN THÀNH ASR Indexing! Tổng cộng {len(asr_index)} đoạn hội thoại đã được nhúng.")