import sys
import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.visual.searcher import VisualSearcher

def evaluate():
    print("Đang tải hệ thống...")
    searcher = VisualSearcher()
    
    # 1. TẠO FILE GROUND TRUTH MẪU (NẾU CHƯA CÓ)
    gt_path = os.path.join("data", "ground_truth.json")
    if not os.path.exists(gt_path):
        sample_gt = {
            "người đàn ông mặc áo đỏ": ["L01_V001_f150.jpg", "L01_V001_crop3.jpg"],
            "chiếc xe máy màu xanh": ["L02_V005_f85.jpg"],
            "bảng hiệu cửa hàng": ["L03_V012_f12.jpg", "L03_V012_crop1.jpg"]
        }
        os.makedirs("data", exist_ok=True)
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(sample_gt, f, indent=4)
        print(f"Đã tạo file mẫu {gt_path}. Hãy điền câu hỏi thật của bạn vào file này rồi chạy lại.")
        return

    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    total_queries = len(ground_truth)
    
    # Biến lưu trữ cho Baseline (Cũ) và Proposed (Mới)
    metrics = {
        "baseline": {"r1": 0, "r5": 0, "r20": 0, "mrr": 0},
        "proposed": {"r1": 0, "r5": 0, "r20": 0, "mrr": 0}
    }

    print(f"\nBắt đầu đánh giá trên {total_queries} câu truy vấn...\n")

    for query, expected_images in ground_truth.items():
        results = searcher.search(query, top_k=50) # Lấy 50 để lọc cho chuẩn
        
        # Tách danh sách kết quả giống như file benchmark_compare.py của bạn
        baseline_results = [res['path'] for res in results if "_crop" not in res['path']]
        proposed_results = [res['path'] for res in results]

        # Hàm tính Rank nội bộ
        def get_rank(retrieved_paths):
            for i, path in enumerate(retrieved_paths[:20]):
                filename = os.path.basename(path)
                for exp in expected_images:
                    # Loại bỏ đuôi .jpg để so sánh tên gốc
                    exp_clean = exp.replace(".jpg", "").replace(".png", "")
                    if exp_clean in filename: # So sánh tên gốc nằm trong tên file kết quả
                        return i + 1
            return -1

        rank_base = get_rank(baseline_results)
        rank_prop = get_rank(proposed_results)

        # Cập nhật điểm Baseline
        if rank_base != -1:
            metrics["baseline"]["mrr"] += 1.0 / rank_base
            if rank_base == 1: metrics["baseline"]["r1"] += 1
            if rank_base <= 5: metrics["baseline"]["r5"] += 1
            if rank_base <= 20: metrics["baseline"]["r20"] += 1

        # Cập nhật điểm Proposed
        if rank_prop != -1:
            metrics["proposed"]["mrr"] += 1.0 / rank_prop
            if rank_prop == 1: metrics["proposed"]["r1"] += 1
            if rank_prop <= 5: metrics["proposed"]["r5"] += 1
            if rank_prop <= 20: metrics["proposed"]["r20"] += 1

    # In kết quả dạng bảng Markdown để copy thẳng vào bài báo
    print("=========================================================================")
    print("| Cấu hình kiến trúc | Recall@1 (%) | Recall@5 (%) | Recall@20 (%) | MRR |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for mode in ["baseline", "proposed"]:
        r1 = (metrics[mode]["r1"] / total_queries) * 100
        r5 = (metrics[mode]["r5"] / total_queries) * 100
        r20 = (metrics[mode]["r20"] / total_queries) * 100
        mrr = metrics[mode]["mrr"] / total_queries
        name = "Baseline (Chỉ Global) " if mode == "baseline" else "Proposed (Global + YOLO)"
        print(f"| {name} | {r1:.2f} | {r5:.2f} | {r20:.2f} | {mrr:.4f} |")
    print("=========================================================================")

if __name__ == "__main__":
    evaluate()