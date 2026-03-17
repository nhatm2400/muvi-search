import json
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from google import genai
from collections import defaultdict
from configs import settings

class FusionSearcher:
    def __init__(self, visual_engine, ocr_engine, asr_engine, api_key):
        self.v = visual_engine
        self.o = ocr_engine
        self.a = asr_engine
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash"

    def analyze_query(self, query):
        """
        GIAI ĐOẠN 1: Orchestrator - Phân tách truy vấn.
        """
        prompt = f"""
        Bạn là chuyên gia phân tích truy vấn video đa phương thức. 
        Hãy tách câu sau thành các thành phần tìm kiếm chi tiết: "{query}"
        
        Trả về DUY NHẤT định dạng JSON với các khóa sau:
        - "visual": Mô tả về hình ảnh, bối cảnh, thực thể hoặc hành động (dành cho SigLIP).
        - "ocr": Các từ khóa văn bản xuất hiện trên màn hình.
        - "ocr_pos": Vị trí dự kiến của văn bản ("logo", "ticker", "center", "global").
        - "speech": Nội dung lời nói hoặc lời thoại (dành cho ASR/PhoBERT).
        - "audio_event": Các âm thanh không phải lời nói như tiếng đàn, tiếng động vật, tiếng nổ (dành cho CLAP).
        
        Nếu một thành phần không có trong câu truy vấn, hãy để chuỗi rỗng.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            text_response = response.text.strip()
            if "```json" in text_response:
                text_response = text_response.split("```json")[1].split("```")[0].strip()
            
            return json.loads(text_response)
        except Exception as e:
            print(f"Lỗi phân tích Gemini: {e}")
            return {
                "visual": query, 
                "ocr": "", 
                "ocr_pos": "global", 
                "speech": "", 
                "audio_event": ""
            }

    def rrf_fusion(self, results_list, k=60):
        """
        GIAI ĐOẠN 3: Hợp nhất kết quả bằng Reciprocal Rank Fusion (RRF).
        """
        fused_scores = defaultdict(float)
        all_metadata = {}

        for engine_res in results_list:
            if not engine_res: continue
            for rank, item in enumerate(engine_res):
                doc_id = item.get('frame_id') or f"{item.get('video_id')}_{item.get('start_time')}"
                
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
                
                if doc_id not in all_metadata:
                    all_metadata[doc_id] = item
                
                all_metadata[doc_id]['fused_score'] = fused_scores[doc_id]

        sorted_results = sorted(all_metadata.values(), key=lambda x: x['fused_score'], reverse=True)
        return sorted_results

    def peak_evaluation(self, results):
        """
        GIAI ĐOẠN 4: Đánh giá khoảnh khắc tiêu biểu.
        """
        if not results: return []
        
        scores = np.array([r.get('fused_score', 0) for r in results])
        
        sigma = getattr(settings, 'PEAK_SIGMA', 1.0)
        smoothed_scores = gaussian_filter1d(scores, sigma=sigma)
        
        for i, res in enumerate(results):
            res['confidence_peak'] = float(smoothed_scores[i])
            
        return sorted(results, key=lambda x: x['confidence_peak'], reverse=True)

    def search(self, query):
        """
        Luồng thực thi chính của Fusion Engine.
        """
        q_parts = self.analyze_query(query)
        
        # Nhánh Visual (Hình ảnh/Hành động)
        res_v = self.v.search(q_parts["visual"], top_k=20)
        
        # Nhánh OCR (Văn bản + Vị trí không gian)
        res_o = []
        if q_parts["ocr"]:
            res_o = self.o.search(q_parts["ocr"], mode=q_parts["ocr_pos"], top_k=15)
            
        # Nhánh ASR Hybrid (Lời nói + Âm thanh môi trường)
        res_a = self.a.search_hybrid(
            speech_query=q_parts["speech"], 
            audio_query=q_parts["audio_event"], 
            top_k=15
        )
        
        fused_results = self.rrf_fusion([res_v, res_o, res_a])
        
        final_results = self.peak_evaluation(fused_results)
        
        return {
            "analysis": q_parts, 
            "results": final_results
        }