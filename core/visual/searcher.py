import os
import json
import torch
import faiss
import numpy as np
import open_clip
from configs import settings

class VisualSearcher:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.MODEL_NAME, pretrained=settings.PRETRAINED_WEIGHTS, device=settings.DEVICE
        )
        self.tokenizer = open_clip.get_tokenizer(settings.MODEL_NAME)

        self.index = faiss.read_index(settings.VISUAL_INDEX_PATH)
        with open(settings.VISUAL_MAP_PATH, 'r') as f:
            self.id_map = json.load(f)

    def search(self, query_text, top_k=20):
        if not query_text: return []
        
        text_tokens = self.tokenizer([query_text]).to(settings.DEVICE)
        with torch.no_grad():
            query_features = self.model.encode_text(text_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_np = query_features.cpu().numpy().astype('float32')

        distances, indices = self.index.search(query_np, top_k)
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if str(idx) in self.id_map:
                rel_path = self.id_map[str(idx)]
                
                web_path = os.path.join("keyframes", rel_path).replace("\\", "/")
                
                results.append({
                    "frame_id": rel_path.split('/')[-1],
                    "path": web_path,
                    "score": float(score)
                })
        return results