# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import base64
from clip_service import CLIPService
import argparse

app = Flask(__name__)

# 全局CLIP服务实例
clip_service = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        topk = int(data.get('topk', 10))
        confidence_threshold = float(data.get('confidence_threshold', 0.0))
        
        if not text:
            return jsonify({'error': '请输入搜索文本'}), 400
        
        # 执行搜索
        results = clip_service.search_images(text, topk=topk, confidence_threshold=confidence_threshold)
        
        # 准备返回数据
        response_data = []
        for result in results:
            response_data.append({
                'image_id': result['image_id'],
                'score': round(result['score'], 4),
                'image_data': f"data:image/jpeg;base64,{result['base64']}"
            })
        
        return jsonify({
            'success': True,
            'results': response_data,
            'count': len(response_data)
        })
        
    except Exception as e:
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'image_count': len(clip_service.image_ids)})

def main():
    parser = argparse.ArgumentParser(description='CLIP Text-to-Image Search Service')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image_data', type=str, required=True, help='Path to image TSV file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on (cuda:0 or cpu)')
    
    args = parser.parse_args()
    
    # 初始化CLIP服务
    global clip_service
    clip_service = CLIPService(
        model_path=args.model_path,
        image_data_path=args.image_data,
        device=args.device
    )
    
    print(f"CLIP服务已启动，共加载 {len(clip_service.image_ids)} 张图片")
    print(f"服务地址: http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()