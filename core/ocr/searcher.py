import os
import pickle
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from Levenshtein import ratio as string_sim_ratio
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from configs import settings

class OCRSearcher:
    def __init__(self):
        with open(settings.OCR_INDEX_PATH, "rb") as f:
            self.offline_index = pickle.load(f)

        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.qwen_processor = AutoProcessor.from_pretrained(model_id)

        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = settings.DEVICE
        self.vietocr_predictor = Predictor(config)
        self.grid_h, self.grid_w = 26, 46 

    def _get_spatial_mask(self, mode):
        mask = np.full((self.grid_h, self.grid_w), 0.1)
        if mode == "ticker": mask[22:, :] = 1.0 
        elif mode == "logo": mask[:5, :12] = 1.0; mask[:5, -12:] = 1.0 
        elif mode == "global": mask[:, :] = 1.0 
        return torch.from_numpy(mask.flatten()).to(settings.DEVICE).float()

    def search(self, query_text, mode="global", top_k=5):
        inputs = self.qwen_processor(text=[query_text], return_tensors="pt").to(settings.DEVICE)
        with torch.no_grad():
            q_out = self.qwen_model(**inputs, output_hidden_states=True)
            query_embeds = F.normalize(q_out.hidden_states[-1].squeeze(0), p=2, dim=-1)

        base_mask = self._get_spatial_mask(mode)
        scored_results = []

        for item in self.offline_index:
            v_embeds = torch.from_numpy(item['embeddings']).to(settings.DEVICE).to(query_embeds.dtype)
            current_mask = F.interpolate(base_mask.view(1, 1, -1), size=v_embeds.shape[0], mode='linear').squeeze() if v_embeds.shape[0] != 1196 else base_mask

            sim = torch.matmul(query_embeds, v_embeds.T) * current_mask.unsqueeze(0)
            max_scores, _ = torch.max(sim, dim=1)
            scored_results.append({"frame_id": item['frame_id'], "raw_score": torch.mean(max_scores).item()})

        top_candidates = sorted(scored_results, key=lambda x: x['raw_score'], reverse=True)[:top_k*3]
        final_moments = []
        for cand in top_candidates:
            video_id = cand['frame_id'].split('_f')[0]
            
            abs_img_path = os.path.join(settings.KEYFRAME_DIR, video_id, cand['frame_id'])
            
            if not os.path.exists(abs_img_path): continue
            
            try:
                detected_text = self.vietocr_predictor.predict(Image.open(abs_img_path))
                text_sim = string_sim_ratio(query_text.lower(), detected_text.lower())
                fusion_score = (cand['raw_score'] * 0.4) + (text_sim * 0.6)
                
                web_path = f"keyframes/{video_id}/{cand['frame_id']}"
                
                final_moments.append({
                    "frame_id": cand['frame_id'], 
                    "path": web_path, 
                    "score": fusion_score, 
                    "text": detected_text
                })
            except Exception as e:
                print(f"Error predicting OCR for {cand['frame_id']}: {e}")
                continue
        
        return sorted(final_moments, key=lambda x: x['score'], reverse=True)[:top_k]