import torch
import faiss
import open_clip
import json
import os
from configs import settings

class VisualSearcher:
    def __init__(self):
        self.model, _, _ = open_clip.create_model_and_transforms(
            settings.MODEL_NAME, pretrained=settings.PRETRAINED_WEIGHTS, device=settings.DEVICE
        )
        self.model.eval()

        wrapper = open_clip.get_tokenizer(settings.MODEL_NAME)
        self.tokenizer = wrapper.tokenizer

        if os.path.exists(settings.VISUAL_INDEX_PATH):
            self.index = faiss.read_index(settings.VISUAL_INDEX_PATH)
        else:
            self.index = None

        if os.path.exists(settings.VISUAL_MAP_PATH):
            with open(settings.VISUAL_MAP_PATH, 'r', encoding='utf-8') as f:
                self.id2path = json.load(f)
        else:
            self.id2path = {}

    def search(self, query_text, top_k=5):
        if not self.index:
            return []

        with torch.no_grad():
            batch = self.tokenizer(
                [query_text],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            input_ids = batch["input_ids"].to(settings.DEVICE)
            
            text_features = self.model.encode_text(input_ids)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            query_vector = text_features.cpu().numpy().astype('float32')

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            str_idx = str(idx)
            if str_idx in self.id2path:
                results.append({
                    "path": self.id2path[str_idx],
                    "score": float(score)
                })

        return results