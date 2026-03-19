import pickle
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, ClapModel, ClapProcessor
from configs import settings

class ASRSearcher:
    def __init__(self):
        print("--- Đang tải dữ liệu ASR (FAISS & Metadata) ---")
        with open(settings.ASR_METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
            
        self.text_index = faiss.read_index(settings.ASR_TEXT_INDEX_PATH)
        self.audio_index = faiss.read_index(settings.ASR_AUDIO_INDEX_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base").to(settings.DEVICE)
        clap_model_id = "laion/clap-htsat-fused"
        self.clap_processor = ClapProcessor.from_pretrained(clap_model_id)
        self.clap_model = ClapModel.from_pretrained(clap_model_id).to(settings.DEVICE)
        self.clap_model.eval()

    def search_hybrid(self, speech_query, audio_query, top_k=10):
        results = []
        
        if speech_query:
            inputs = self.tokenizer(speech_query, return_tensors="pt", padding=True, truncation=True, max_length=256).to(settings.DEVICE)
            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
                query_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
                faiss.normalize_L2(query_vec) 
                
            distances, indices = self.text_index.search(query_vec, top_k)
            for score, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    meta = self.metadata[idx]
                    vid_id = meta.get('video_id') or meta.get('vid_name') or meta.get('id') or "unknown_video"
                    start_t = meta.get('start_time', meta.get('start', 0.0))
                    
                    results.append({
                        "video_id": vid_id,
                        "start_time": start_t,
                        "text": meta.get('text', ''),
                        "score": float(score),
                        "type": "speech"
                    })
                    
        if audio_query:
            inputs = self.clap_processor(text=[audio_query], return_tensors="pt").to(settings.DEVICE)
            with torch.no_grad():
                query_vec = self.clap_model.get_text_features(**inputs).cpu().numpy().astype('float32')
                faiss.normalize_L2(query_vec)
                
            distances, indices = self.audio_index.search(query_vec, top_k)
            for score, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    meta = self.metadata[idx]
                    vid_id = meta.get('video_id') or meta.get('vid_name') or meta.get('id') or "unknown_video"
                    start_t = meta.get('start_time', meta.get('start', 0.0))
                    
                    results.append({
                        "video_id": vid_id,
                        "start_time": start_t,
                        "text": f"Âm thanh: {audio_query}",
                        "score": float(score),
                        "type": "audio_event"
                    })

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:top_k]