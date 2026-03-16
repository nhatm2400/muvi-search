import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cv2
import torch
import open_clip
import faiss
import numpy as np
import json
import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from configs import settings

class KeyframeDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # Nên convert sang RGB để tránh lỗi với ảnh grayscale hoặc RGBA
            img = Image.open(path).convert('RGB')
            return self.preprocess(img), path
        except Exception as e:
            # Trả về tensor rỗng và cờ lỗi để filter ở DataLoader
            return torch.zeros(3, 224, 224), "ERROR"

class VisualIndexer:
    def __init__(self):
        self.processed_videos = set()
        os.makedirs(os.path.dirname(settings.PROCESSED_LOG_PATH), exist_ok=True)
        
        if os.path.exists(settings.PROCESSED_LOG_PATH):
            with open(settings.PROCESSED_LOG_PATH, "r", encoding="utf-8") as f:
                self.processed_videos = set(line.strip() for line in f if line.strip())

    def extract_keyframes(self, video_path):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        video_subfolder = os.path.join(settings.KEYFRAME_DIR, filename)

        if os.path.exists(video_subfolder) and len(os.listdir(video_subfolder)) > 0:
            return 0 
        
        os.makedirs(video_subfolder, exist_ok=True)

        cap = None
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=27.0))
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list()

            cap = cv2.VideoCapture(video_path)
            count = 0
            
            # Dùng set để tự động loại bỏ các frame bị trùng lặp (nếu có)
            frames_to_cut_set = set()
            
            if not scene_list:
                frames_to_cut_set.add(0)
            else:
                for scene in scene_list:
                    start_tc = scene[0]
                    end_tc = scene[1]
                    
                    start_frame = start_tc.get_frames()
                    end_frame = end_tc.get_frames()
                    
                    # Tính thời lượng của scene bằng giây
                    duration_sec = end_tc.get_seconds() - start_tc.get_seconds()
                    
                    # Frame ở giữa (luôn lấy)
                    mid_frame = start_frame + (end_frame - start_frame) // 2
                    
                    if duration_sec < 3.0:
                        # Scene ngắn (< 3 giây): Chỉ lấy 1 frame ở giữa
                        frames_to_cut_set.add(mid_frame)
                    else:
                        # Scene dài (>= 3 giây): Lấy 3 frame Đầu - Giữa - Cuối
                        # Trừ 1 ở frame cuối để không lẹm sang scene tiếp theo
                        last_frame = max(start_frame, end_frame - 1)
                        frames_to_cut_set.add(start_frame)
                        frames_to_cut_set.add(mid_frame)
                        frames_to_cut_set.add(last_frame)

            # Sắp xếp lại danh sách frame từ nhỏ đến lớn để cv2.VideoCapture đọc mượt hơn
            frames_to_cut = sorted(list(frames_to_cut_set))

            for frame_idx in frames_to_cut:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    out_path = os.path.join(video_subfolder, f"{filename}_f{frame_idx}.jpg")
                    cv2.imwrite(out_path, frame)
                    count += 1
            
            return count

        except Exception as e:
            print(f"[ERROR] Processing {filename}: {e}")
            return 0
        finally:
            if cap is not None:
                cap.release()

    def run_extraction(self):
        if not os.path.exists(settings.VIDEO_DIR):
            print(f"Error: Video directory not found at {settings.VIDEO_DIR}")
            return

        video_files = [f for f in os.listdir(settings.VIDEO_DIR) if f.endswith(('.mp4', '.mkv', '.avi'))]
        print(f"Found {len(video_files)} videos. Starting extraction...")

        total_extracted = 0
        for vid in tqdm(video_files, desc="Smart Extraction"):
            if vid in self.processed_videos:
                continue
            
            full_path = os.path.join(settings.VIDEO_DIR, vid)
            num = self.extract_keyframes(full_path)
            total_extracted += num
            
            with open(settings.PROCESSED_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(vid + "\n")
            
            self.processed_videos.add(vid) # Cập nhật luôn set in-memory
        
        print(f"Extraction finished. Total new frames: {total_extracted}")

    def run_indexing(self):
        print(f"Loading Model {settings.MODEL_NAME} on {settings.DEVICE}...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.MODEL_NAME, pretrained=settings.PRETRAINED_WEIGHTS, device=settings.DEVICE
        )
        model.eval()

        search_pattern = os.path.join(settings.KEYFRAME_DIR, "**", "*.jpg")
        image_paths = sorted(glob.glob(search_pattern, recursive=True))
        print(f"Found {len(image_paths)} keyframes to index.")

        if len(image_paths) == 0:
            print("No keyframes found! Please run extraction first.")
            return

        dataset = KeyframeDataset(image_paths, preprocess)
        dataloader = DataLoader(
            dataset, 
            batch_size=settings.BATCH_SIZE, 
            num_workers=settings.NUM_WORKERS, 
            pin_memory=True,
            shuffle=False
        )

        features_list = []
        id2path = {}
        global_idx = 0

        print("Starting embedding process...")
        with torch.no_grad():
            for batch_images, batch_paths in tqdm(dataloader, desc="Embedding"):
                # SỬA LỖI: mask phải là tensor kiểu bool để index được trên batch_images
                valid_mask = torch.tensor([p != "ERROR" for p in batch_paths], dtype=torch.bool)
                if not valid_mask.any(): 
                    continue
                
                real_images = batch_images[valid_mask].to(settings.DEVICE)
                
                # TỐI ƯU: Sử dụng autocast để tăng tốc inference (nếu dùng GPU)
                with torch.autocast(device_type=settings.DEVICE.split(':')[0], enabled=(settings.DEVICE != 'cpu')):
                    batch_features = model.encode_image(real_images)
                
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
                features_list.append(batch_features.cpu().numpy().astype('float32'))
                
                # Lọc ra các path hợp lệ
                valid_paths = [p for p in batch_paths if p != "ERROR"]
                for path in valid_paths:
                    # TỐI ƯU: Bỏ try/except thừa, dùng os.path thống nhất
                    rel_path = os.path.relpath(path, settings.KEYFRAME_DIR).replace("\\", "/")
                    id2path[str(global_idx)] = rel_path
                    global_idx += 1

        if features_list:
            all_features = np.concatenate(features_list)
            
            index = faiss.IndexFlatIP(all_features.shape[1])
            index.add(all_features)
            
            faiss.write_index(index, settings.VISUAL_INDEX_PATH)
            with open(settings.VISUAL_MAP_PATH, 'w', encoding='utf-8') as f:
                json.dump(id2path, f, indent=4)
                
            print(f"Success! Saved Index to: {settings.VISUAL_INDEX_PATH}")
            print(f"Success! Saved Mapping to: {settings.VISUAL_MAP_PATH}")
        else:
            print("No features were extracted.")