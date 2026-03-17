"""
=============================================================================
MUVI-SEARCH INDEXING PIPELINE
=============================================================================
Hướng dẫn chạy script:

1. Chạy toàn bộ hệ thống (Visual, OCR, ASR):
   python scripts/run_indexing.py --mode all

2. Chạy riêng từng phương thức:
   - Visual: python scripts/run_indexing.py --mode visual
   - OCR:    python scripts/run_indexing.py --mode ocr
   - ASR:    python scripts/run_indexing.py --mode asr

3. Chạy từng bước trong Visual Pipeline (dùng thêm tham số --step):
   - Chỉ trích xuất frame: python scripts/run_indexing.py --mode visual --step extract
   - Chỉ crop vật thể:     python scripts/run_indexing.py --mode visual --step crop
   - Chỉ nhúng vector:     python scripts/run_indexing.py --mode visual --step index
=============================================================================
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual.indexer import VisualIndexer
from core.ocr.indexer import OCRIndexer
from core.asr.indexer import ASRIndexer
from core.visual.cropper import main as run_yolo_cropping

def main():
    parser = argparse.ArgumentParser(description="Muvi-Search Multi-modal Indexing Pipeline")
    parser.add_argument('--mode', type=str, choices=['visual', 'ocr', 'asr', 'all'], required=True)
    parser.add_argument('--step', type=str, choices=['extract', 'crop', 'index', 'all'], default='all')
    
    args = parser.parse_args()

    if args.mode in ['visual', 'all']:
        print(f"\n=== VISUAL PIPELINE (Chế độ: {args.step.upper()}) ===")
        indexer = VisualIndexer()
        
        if args.step in ['extract', 'all']:
            indexer.run_extraction()    
        
        if args.step in ['crop', 'all']:
            run_yolo_cropping()

        if args.step in ['index', 'all']:
            indexer.run_indexing()

    if args.mode in ['ocr', 'all']:
        print("\n=== OCR PIPELINE (Qwen2-VL) ===")
        o_indexer = OCRIndexer()
        o_indexer.build_index()

    if args.mode in ['asr', 'all']:
        print("\n=== ASR PIPELINE (Whisper + PhoBERT) ===")
        a_indexer = ASRIndexer()
        a_indexer.build_index()

    print("\n[+] TOÀN BỘ QUÁ TRÌNH INDEXING ĐÃ HOÀN TẤT!")

if __name__ == "__main__":
    main()