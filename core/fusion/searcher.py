import json
import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from groq import Groq
from collections import defaultdict
from configs import settings

class FusionSearcher:
    def __init__(self, visual_engine, ocr_engine, asr_engine, api_key):
        self.v = visual_engine
        self.o = ocr_engine
        self.a = asr_engine
        
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"

    def analyze_query(self, query):
        """
        GIAI ĐOẠN 1: Orchestrator - Phân tách truy vấn bằng Groq (Llama 3).
        """
        prompt = f"""
        Bạn là hệ thống bóc tách thông tin tìm kiếm video đa phương thức.
        Nhiệm vụ: Phân tách câu truy vấn "{query}" thành các thành phần cụ thể.
        
        QUY TẮC TUYỆT ĐỐI: Trích xuất nguyên văn. KHÔNG ĐƯỢC TÓM TẮT. Xóa bỏ các từ nối không cần thiết.

        Trả về DUY NHẤT định dạng JSON với các khóa:
        - "visual": Cụm từ miêu tả hình ảnh, màu sắc, hành động, bối cảnh. (LƯU Ý: Phải loại bỏ các phần liên quan đến chữ viết hoặc âm thanh ra khỏi đây).
        - "ocr": Nội dung chữ viết, văn bản xuất hiện trên màn hình. (Dấu hiệu nhận biết: đi ngay sau các từ "chữ", "dòng chữ", "biển báo", hoặc nằm trong ngoặc kép). Ví dụ: truy vấn "dòng chữ Hồ Chí Minh" -> ocr là "Hồ Chí Minh".
        - "ocr_pos": Vị trí dự kiến của văn bản ("logo", "ticker", "center", "global").
        - "speech": Nội dung lời nói, lời thoại. (Dấu hiệu: đi sau từ "nói", "kể", "thoại").
        - "audio_event": Âm thanh môi trường. (Dấu hiệu: đi sau từ "tiếng", "âm thanh" như "tiếng chim", "tiếng súng").
        
        Nếu một thành phần không có, BẮT BUỘC để chuỗi rỗng "".
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1 
            )
            
            text_response = response.choices[0].message.content.strip()
            return json.loads(text_response)
            
        except Exception as e:
            print(f"[!] Lỗi phân tích Groq: {e}. Đang kích hoạt Heuristic Fallback (Regex)...")
            fallback = { "visual": query, "ocr": "", "ocr_pos": "global", "speech": "", "audio_event": "" }
            query_lower = query.lower()
            
            ocr_match = re.search(r"['\"](.*?)['\"]", query)
            if ocr_match:
                fallback["ocr"] = ocr_match.group(1)
                fallback["visual"] = query.replace(ocr_match.group(0), "").strip()
            
            if "tiếng" in query_lower or "âm thanh" in query_lower or "nghe" in query_lower:
                fallback["audio_event"] = query
                fallback["visual"] = re.sub(r'(?i)(tiếng|âm thanh|nghe).*', '', fallback["visual"]).strip()
            elif "nói" in query_lower or "kể" in query_lower or "thoại" in query_lower:
                fallback["speech"] = query
                
            return fallback

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
        
        res_v = self.v.search(q_parts["visual"], top_k=20)
        res_o = []
        if q_parts["ocr"]:
            res_o = self.o.search(q_parts["ocr"], mode=q_parts["ocr_pos"], top_k=15)
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