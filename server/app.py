import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from flask import Flask, render_template, request, jsonify, send_from_directory
from configs import settings
from core.visual.searcher import VisualSearcher
from core.ocr.searcher import OCRSearcher

app = Flask(__name__, template_folder=os.path.join(project_root, 'templates'))

print("Server starting... Loading Engines...")
visual_engine = VisualSearcher()
ocr_engine = OCRSearcher()
print("All engines loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(settings.KEYFRAME_DIR, filename)

@app.route('/api/search/visual')
def search_visual():
    query = request.args.get('q', '').strip()
    top_k = int(request.args.get('k', 20))
    
    if not query:
        return jsonify([])

    print(f"Visual Searching for: '{query}'")
    results = visual_engine.search(query, top_k=top_k)
    
    response = []
    for item in results:
        web_path = item['path'].replace('\\', '/')
        response.append({
            "image_url": f"/images/{web_path}",
            "score": round(item['score'], 4),
            "filename": os.path.basename(item['path'])
        })
        
    return jsonify(response)

@app.route('/api/search/ocr')
def search_ocr():
    query = request.args.get('q', '').strip()
    mode = request.args.get('mode', 'global').strip()
    top_k_s1 = int(request.args.get('k_s1', 10))
    top_k_final = int(request.args.get('k_final', 5))
    
    if not query:
        return jsonify([])

    print(f"OCR Searching for: '{query}' | Mode: {mode}")
    results = ocr_engine.search(query, mode=mode, top_k_s1=top_k_s1, top_k_final=top_k_final)
    
    response = []
    for item in results:
        rel_path = os.path.relpath(item['path'], settings.KEYFRAME_DIR)
        web_path = rel_path.replace('\\', '/')
        
        response.append({
            "image_url": f"/images/{web_path}",
            "score": round(item['confidence'], 4),
            "detected_text": item['detected_text'],
            "filename": item['frame_id'],
            "timestamp": item['timestamp']
        })
        
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)