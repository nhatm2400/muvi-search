import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual.indexer import VisualIndexer

def main():
    print("=== MVS PROJECT: OFFLINE INDEXING PIPELINE ===")
    
    indexer = VisualIndexer()
    
    print("\n--- STEP 1: KEYFRAME EXTRACTION ---")
    indexer.run_extraction()    
    print("\n--- STEP 2: VECTOR EMBEDDING & INDEXING ---")
    indexer.run_indexing()
    
    print("\n=== PIPELINE COMPLETED ===")

if __name__ == "__main__":
    main()