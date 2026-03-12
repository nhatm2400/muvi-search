import os

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_pir_force_legacy_interpreter"] = "1"

import pickle
import torch
import numpy as np
import re
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
from Levenshtein import ratio as string_sim_ratio

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PaddleOCR

from configs import settings

class OCRSearcher:
    def __init__(self):
        print("Initializing OCR Searcher (Cascaded Pipeline)...")

        self.index_path = settings.OCR_INDEX_PATH
        class Numpy1xUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module, name)

        with open(self.index_path, "rb") as f:
            self.offline_index = Numpy1xUnpickler(f).load()

        print("Loading Qwen2-VL-2B-Instruct...")
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.qwen_processor = AutoProcessor.from_pretrained(model_id)

        print("Loading DBNet++ Detector...")
        self.dbnet_detector = PaddleOCR(
            use_angle_cls=False, 
            lang='vi', 
            show_log=False, 
            use_gpu=True,
            ir_optim=False      
        ) 

        print("Loading VietOCR model...")
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = settings.DEVICE
        config['predictor']['beamsearch'] = True
        self.vietocr_predictor = Predictor(config)

        self.grid_h = 26
        self.grid_w = 46
        print("OCR Searcher Ready!")

    def get_timestamp_from_filename(self, filename, default_fps=25):
        try:
            match = re.search(r'(L\d+_V\d+)_f(\d+)', filename)
            if match:
                video_id = match.group(1)
                frame_idx = int(match.group(2))
                return {"sec": round(frame_idx / default_fps, 2), "video_id": video_id, "frame_idx": frame_idx}
        except: pass
        return {"sec": 0.0, "video_id": "unknown", "frame_idx": 0}

    def stage1_coarse_retrieval(self, query_text, mode="global", threshold=0.3):
        inputs = self.qwen_processor(text=[query_text], return_tensors="pt").to(settings.DEVICE)
        with torch.no_grad():
            q_out = self.qwen_model(**inputs, output_hidden_states=True)
            query_embeds = F.normalize(q_out.hidden_states[-1].squeeze(0), p=2, dim=-1)

        mask = np.full((self.grid_h, self.grid_w), 0.1)
        if mode == "ticker": mask[22:, :] = 1.0
        elif mode == "logo": mask[:5, :12] = 1.0; mask[:5, -12:] = 1.0
        elif mode == "global": mask[:, :] = 1.0
        spatial_mask = torch.from_numpy(mask.flatten()).to(settings.DEVICE).float()

        scored_results = []
        for item in self.offline_index:
            v_embeds = torch.from_numpy(item['embeddings']).to(settings.DEVICE).to(query_embeds.dtype)
            
            if v_embeds.shape[0] > 1196: v_embeds = v_embeds[:1196]

            sim_matrix = torch.matmul(query_embeds, v_embeds.T)
            weighted_sim = sim_matrix * spatial_mask.unsqueeze(0)
            
            max_scores, _ = torch.max(weighted_sim, dim=1)
            k = max(1, int(max_scores.shape[0] * 0.2))
            top_k_scores, _ = torch.topk(max_scores, k)

            scored_results.append({"frame_id": item['frame_id'], "raw_score": torch.mean(top_k_scores).item()})

        scores = np.array([f['raw_score'] for f in scored_results])
        smoothed_scores = gaussian_filter1d(scores, sigma=1.0)
        for i, f in enumerate(scored_results): f['smoothed_score'] = smoothed_scores[i]

        return sorted([f for f in scored_results if f['smoothed_score'] > threshold],
                      key=lambda x: x['smoothed_score'], reverse=True)

    def stage2_dbnet_vietocr_reranking(self, query_text, top_frames, mode="global", top_k=5):
        final_moments = []
        for cand in top_frames[:top_k]:
            meta = self.get_timestamp_from_filename(cand['frame_id'])
            img_path = os.path.join(settings.KEYFRAME_DIR, meta['video_id'], cand['frame_id'])
            if not os.path.exists(img_path): img_path = os.path.join(settings.KEYFRAME_DIR, cand['frame_id'])

            try:
                result = self.dbnet_detector.ocr(img_path, rec=False)

                best_text_score = 0.0
                display_text = "<Không phát hiện chữ>"
                if result and result[0]:
                    img_pil = Image.open(img_path).convert("RGB")
                    img_w, img_h = img_pil.size
                    texts = []
                    
                    for line in result[0]:
                        box = np.array(line).astype(np.int32)
                        
                        y_min, y_max = max(0, box[:,1].min()), box[:,1].max()
                        x_min, x_max = max(0, box[:,0].min()), box[:,0].max()

                        if mode == "ticker":
                            if y_min < img_h * 0.75: 
                                continue 
                        elif mode == "logo":
                            if y_max > img_h * 0.25: 
                                continue 
                            if x_min > img_w * 0.25 and x_max < img_w * 0.75:
                                continue

                        pad = 5
                        roi_y_min, roi_y_max = max(0, y_min - pad), min(img_h, y_max + pad)
                        roi_x_min, roi_x_max = max(0, x_min - pad), min(img_w, x_max + pad)
                        roi = img_pil.crop((roi_x_min, roi_y_min, roi_x_max, roi_y_max))
                        
                        text = self.vietocr_predictor.predict(roi)
                        if text.strip():
                            texts.append(text)
                            
                            query_lower = query_text.lower()
                            text_lower = text.lower()
                            
                            if query_lower in text_lower:
                                sim = 1.0
                            else:
                                sim = string_sim_ratio(query_lower, text_lower)
                                
                            if sim > best_text_score: best_text_score = sim
                    if texts: display_text = " | ".join(texts)

                final_score = (cand['smoothed_score'] * 0.4 + best_text_score * 0.6)
                final_moments.append({"timestamp": meta, "frame_id": cand['frame_id'], 
                                     "detected_text": display_text, "confidence": final_score, "path": img_path})
            except Exception as e:
                print(f"Lỗi {cand['frame_id']}: {e}"); continue
        return sorted(final_moments, key=lambda x: x['confidence'], reverse=True)

    def search(self, query_text, mode="global", top_k_s1=50, top_k_final=3):
        res_s1 = self.stage1_coarse_retrieval(query_text, mode=mode)
        if not res_s1: return []
        return self.stage2_dbnet_vietocr_reranking(query_text, res_s1, mode=mode, top_k=top_k_s1)[:top_k_final]