import sys
import os
import torch
import open_clip
import faiss
import json
import matplotlib.pyplot as plt
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from configs import settings

QUERY_TEXT = "a man and a woman" 
TOP_K = 5

def safe_tokenize(text_list, tokenizer_obj, context_length=64):
    batch = tokenizer_obj(
        text_list,
        padding="max_length",
        truncation=True,
        max_length=context_length,
        return_tensors="pt"
    )
    return batch["input_ids"]

def main():
    print(f"Running Test on Device: {settings.DEVICE}")
    print(f"Index File: {settings.VISUAL_INDEX_PATH}")
    
    print("Loading SigLIP Model...")
    model, _, _ = open_clip.create_model_and_transforms(
        settings.MODEL_NAME, 
        pretrained=settings.PRETRAINED_WEIGHTS, 
        device=settings.DEVICE
    )
    model.eval()

    wrapper = open_clip.get_tokenizer(settings.MODEL_NAME)
    hf_tokenizer = wrapper.tokenizer

    if not os.path.exists(settings.VISUAL_INDEX_PATH):
        print(f"Error: Không tìm thấy file Index tại {settings.VISUAL_INDEX_PATH}")
        return

    print("Loading FAISS Index...")
    index = faiss.read_index(settings.VISUAL_INDEX_PATH)
    
    with open(settings.VISUAL_MAP_PATH, 'r', encoding='utf-8') as f:
        id2path = json.load(f)

    print(f"Searching for: '{QUERY_TEXT}'")

    with torch.no_grad():
        text = safe_tokenize([QUERY_TEXT], hf_tokenizer).to(settings.DEVICE)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        text_vector = text_features.cpu().numpy().astype('float32')
        distances, indices = index.search(text_vector, TOP_K)

    plt.figure(figsize=(15, 5))
    print(f"\n--- Top {TOP_K} Results ---")

    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1: continue
        
        str_idx = str(idx)
        if str_idx in id2path:
            rel_path = id2path[str_idx]
            full_path = os.path.join(settings.KEYFRAME_DIR, rel_path)
            
            print(f"#{i+1}: Score {dist:.4f} - {rel_path}")
            
            try:
                img = Image.open(full_path)
                plt.subplot(1, TOP_K, i+1)
                plt.imshow(img)
                plt.title(f"Score: {dist:.2f}\n{os.path.basename(rel_path)}")
                plt.axis('off')
            except Exception as e:
                print(f"Cannot read image: {full_path}")
        else:
            print(f"Index {idx} not found in JSON mapping.")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()