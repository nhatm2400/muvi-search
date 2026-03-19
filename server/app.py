import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from core.visual.searcher import VisualSearcher
from core.ocr.searcher import OCRSearcher
from core.asr.searcher import ASRSearcher
from core.fusion.searcher import FusionSearcher

app = Flask(__name__, template_folder='../templates')

DATA_PATH = os.path.join(ROOT_DIR, "data")

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DATA_PATH, filename)

print("--- Đang khởi tạo hệ thống tìm kiếm đa phương thức ---")
v_engine = VisualSearcher()
o_engine = OCRSearcher()
a_engine = ASRSearcher()
fusion_engine = FusionSearcher(v_engine, o_engine, a_engine, api_key=os.getenv("GROQ_API_KEY", ""))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search/fusion', methods=['POST'])
def search_fusion():
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "Truy vấn không được để trống"}), 400
        
    try:
        output = fusion_engine.search(query)
        return jsonify(output)
    except Exception as e:
        print(f"Lỗi xử lý tìm kiếm: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)