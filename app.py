import os
import io
import time
import json
import base64
import urllib.request
import urllib.error
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload limit

DEFAULT_ENDPOINT = "https://predict-6afb3d9aba07484915b0492b-dproatj77a-nw.a.run.app/predict"
HEALTH_ENDPOINT = "https://predict-6afb3d9aba07484915b0492b-dproatj77a-nw.a.run.app/health"
ROOT_ENDPOINT = "https://predict-6afb3d9aba07484915b0492b-dproatj77a-nw.a.run.app/"

TARGET_IDENTITY = "Caleb"


def call_ultralytics_cloud(image_bytes: bytes, filename: str, api_key: str, conf: float = 0.25, iou: float = 0.45):
    """
    Sends image bytes to Ultralytics Dedicated Cloud Run Endpoint with multipart/form-data.
    Uses standard library urllib so there are zero third-party networking issues.
    """
    boundary = '----WebKitFormBoundary' + hex(int(time.time() * 1000))[2:]
    
    parts = []
    
    # 1. file part
    parts.append(
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'.encode('utf-8')
        + image_bytes + b'\r\n'
    )
    
    # 2. conf part
    parts.append(
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="conf"\r\n\r\n'
        f'{conf}\r\n'.encode('utf-8')
    )
    
    # 3. iou part
    parts.append(
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="iou"\r\n\r\n'
        f'{iou}\r\n'.encode('utf-8')
    )
    
    # End boundary
    parts.append(f'--{boundary}--\r\n'.encode('utf-8'))
    
    body = b''.join(parts)
    
    headers = {
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Content-Length': str(len(body)),
        'User-Agent': 'NeuraFace-Client/1.0'
    }
    
    if api_key:
        clean_key = api_key.strip()
        if not clean_key.startswith('Bearer '):
            clean_key = f'Bearer {clean_key}'
        headers['Authorization'] = clean_key
        
    req = urllib.request.Request(DEFAULT_ENDPOINT, data=body, headers=headers, method='POST')
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            raw_data = resp.read().decode('utf-8')
            return True, json.loads(raw_data), elapsed_ms, 200
    except urllib.error.HTTPError as e:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        err_body = e.read().decode('utf-8', errors='ignore')
        try:
            err_json = json.loads(err_body)
            msg = err_json.get('message', err_json.get('error', err_body))
        except Exception:
            msg = err_body
        return False, msg, elapsed_ms, e.code
    except Exception as e:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        return False, str(e), elapsed_ms, 500


def create_biometric_overlay(pil_img: Image.Image, detections: list):
    """
    Draws high-tech cyan/teal biometric reticles and identity labels on the image.
    Returns: (annotated_base64_data_url, list_of_face_crops_with_metadata)
    """
    # Work on a copy in RGBA for crisp alpha-blending
    annotated = pil_img.convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    
    # Try loading a readable font, fallback to default
    try:
        font_large = ImageFont.truetype("arial.ttf", max(16, int(w * 0.026)))
        font_small = ImageFont.truetype("arial.ttf", max(12, int(w * 0.019)))
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
    crops = []
    
    for idx, det in enumerate(detections):
        box = det.get('box', {})
        x1 = max(0, min(w, float(box.get('x1', 0))))
        y1 = max(0, min(h, float(box.get('y1', 0))))
        x2 = max(0, min(w, float(box.get('x2', 0))))
        y2 = max(0, min(h, float(box.get('y2', 0))))
        
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue
            
        conf = float(det.get('confidence', 0))
        name = str(det.get('name', 'Face')).strip()
        is_target = TARGET_IDENTITY.lower() in name.lower()
        
        # Color definitions: Laser Cyan & Electric Teal
        # Caleb = Electric Cyan (#06B6D4), other = Bright Amber / Mint
        primary_color = (6, 182, 212, 255) if is_target else (45, 212, 191, 255)
        accent_color = (45, 212, 191, 255)
        glow_fill = (6, 182, 212, 35) if is_target else (45, 212, 191, 25)
        
        # 1. Subtle semi-transparent inner tint
        draw.rectangle([x1, y1, x2, y2], fill=glow_fill)
        
        # 2. Thin bounding box border
        draw.rectangle([x1, y1, x2, y2], outline=(primary_color[0], primary_color[1], primary_color[2], 120), width=1)
        
        # 3. Biometric Corner Reticles (Tactical brackets on 4 corners)
        corner_len = max(8, int(min(bw, bh) * 0.22))
        corner_width = max(2, int(w * 0.004))
        
        # Top-Left
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=primary_color, width=corner_width)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=primary_color, width=corner_width)
        
        # Top-Right
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=primary_color, width=corner_width)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=primary_color, width=corner_width)
        
        # Bottom-Left
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=primary_color, width=corner_width)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=primary_color, width=corner_width)
        
        # Bottom-Right
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=primary_color, width=corner_width)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=primary_color, width=corner_width)
        
        # 4. Center Micro-Crosshair
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ch_size = max(4, int(min(bw, bh) * 0.06))
        draw.line([(cx - ch_size, cy), (cx + ch_size, cy)], fill=accent_color, width=1)
        draw.line([(cx, cy - ch_size), (cx, cy + ch_size)], fill=accent_color, width=1)
        
        # 5. High-Tech Badge on top of box
        label_text = f"● {name.upper()}" if is_target else f"● {name}"
        score_text = f"{conf * 100:.1f}% MATCH"
        
        # Badge background
        badge_h = max(22, int(bh * 0.18))
        badge_w = max(110, int(bw * 0.85))
        badge_y = max(0, y1 - badge_h - 4)
        
        draw.rectangle([x1, badge_y, x1 + badge_w, badge_y + badge_h], fill=(6, 16, 23, 230))
        draw.rectangle([x1, badge_y, x1 + badge_w, badge_y + badge_h], outline=primary_color, width=1)
        
        # Draw texts
        draw.text((x1 + 6, badge_y + 3), f"{label_text} | {score_text}", fill=primary_color, font=font_small)
        
        # 6. Extract Face Crop with padding for the biometric gallery
        pad_x = bw * 0.15
        pad_y = bh * 0.20
        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(w, int(x2 + pad_x))
        crop_y2 = min(h, int(y2 + pad_y))
        
        face_crop = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        # Resize thumbnail to square 180x180
        thumb_io = io.BytesIO()
        face_crop.resize((200, 200), Image.Resampling.LANCZOS).convert("RGB").save(thumb_io, format='JPEG', quality=90)
        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_io.getvalue()).decode('utf-8')
        
        crops.append({
            'index': idx + 1,
            'name': name,
            'is_caleb': is_target,
            'confidence': conf,
            'confidence_pct': round(conf * 100, 1),
            'box': {'x1': round(x1, 1), 'y1': round(y1, 1), 'x2': round(x2, 1), 'y2': round(y2, 1)},
            'box_norm': {
                'x1': round(x1 / w, 4),
                'y1': round(y1 / h, 4),
                'x2': round(x2 / w, 4),
                'y2': round(y2 / h, 4)
            },
            'thumbnail': crop_b64
        })
        
    # Convert annotated back to RGB JPEG base64
    out_io = io.BytesIO()
    annotated.convert("RGB").save(out_io, format='JPEG', quality=92)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(out_io.getvalue()).decode('utf-8')
    
    return annotated_b64, crops


@app.route('/')
def index():
    return render_template('index.html', default_endpoint=DEFAULT_ENDPOINT, target_identity=TARGET_IDENTITY)


@app.route('/api/status', methods=['GET'])
def get_status():
    """Pings the Cloud Run health endpoint to verify connectivity."""
    t0 = time.time()
    try:
        req = urllib.request.Request(HEALTH_ENDPOINT, headers={'User-Agent': 'NeuraFace-Client/1.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            ms = round((time.time() - t0) * 1000, 1)
            return jsonify({
                'online': True,
                'status': data.get('status', 'healthy'),
                'latency_ms': ms,
                'endpoint': DEFAULT_ENDPOINT,
                'message': 'Ultralytics Dedicated Cloud Run is live and healthy'
            })
    except Exception as e:
        ms = round((time.time() - t0) * 1000, 1)
        return jsonify({
            'online': False,
            'status': 'offline',
            'latency_ms': ms,
            'endpoint': DEFAULT_ENDPOINT,
            'error': str(e)
        }), 502


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Receives image from frontend (file upload or base64 webcam snapshot),
    forwards it to Ultralytics Dedicated Endpoint, parses detections,
    generates annotated biometrics and crops, and returns complete payload.
    """
    # 1. Extract API Key (from header, form, or env var)
    api_key = request.headers.get('X-API-Key') or request.form.get('api_key') or os.environ.get('ULTRALYTICS_API_KEY', '')
    api_key = api_key.strip()
    
    if not api_key:
        return jsonify({
            'success': False,
            'error_type': 'AUTH_REQUIRED',
            'error': 'API Key Required',
            'message': 'Please enter your Ultralytics Platform API Key (ul_xxx) in the top settings bar to run inference.'
        }), 401

    # 2. Extract Confidence and IoU parameters
    try:
        conf = float(request.form.get('conf', 0.25))
    except (ValueError, TypeError):
        conf = 0.25
        
    try:
        iou = float(request.form.get('iou', 0.45))
    except (ValueError, TypeError):
        iou = 0.45

    # 3. Extract Image (either multipart file or base64 data)
    image_bytes = None
    filename = "upload.jpg"
    
    if 'file' in request.files and request.files['file'].filename:
        uploaded_file = request.files['file']
        filename = uploaded_file.filename
        image_bytes = uploaded_file.read()
    elif 'image_b64' in request.form and request.form['image_b64']:
        # Data URL format: data:image/jpeg;base64,...
        b64_str = request.form['image_b64']
        if ',' in b64_str:
            b64_str = b64_str.split(',', 1)[1]
        try:
            image_bytes = base64.b64decode(b64_str)
            filename = "webcam_capture.jpg"
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to decode base64 image: {str(e)}'}), 400
            
    if not image_bytes:
        return jsonify({'success': False, 'error': 'No image provided. Please upload a photo or capture via webcam.'}), 400

    # 4. Load and validate image locally with Pillow
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert("RGB")
        orig_w, orig_h = pil_img.size
        
        # Prepare original as base64 for side-by-side inspection
        orig_io = io.BytesIO()
        pil_img.save(orig_io, format='JPEG', quality=88)
        orig_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_io.getvalue()).decode('utf-8')
    except Exception as e:
        return jsonify({'success': False, 'error': f'Invalid image format: {str(e)}'}), 400

    # 5. Call Ultralytics Cloud Run Endpoint
    ok, result_data, elapsed_ms, status_code = call_ultralytics_cloud(
        image_bytes=image_bytes,
        filename=filename,
        api_key=api_key,
        conf=conf,
        iou=iou
    )
    
    if not ok:
        if status_code == 401:
            return jsonify({
                'success': False,
                'error_type': 'AUTH_INVALID',
                'error': 'Invalid API Key',
                'message': 'Ultralytics returned 401 Unauthorized. Double-check your Platform API Key (it must start with ul_).'
            }), 401
        return jsonify({
            'success': False,
            'error_type': 'CLOUD_ERROR',
            'error': f'Ultralytics API Error ({status_code})',
            'message': str(result_data)
        }), status_code

    # 6. Parse Ultralytics Prediction Response
    # Format expected: {"images": [{"shape": [...], "results": [{"name": "Caleb", "confidence": 0.92, "box": {...}}]}]}
    detections = []
    try:
        images_arr = result_data.get('images', [])
        if images_arr and len(images_arr) > 0:
            detections = images_arr[0].get('results', [])
        elif 'results' in result_data:
            detections = result_data.get('results', [])
    except Exception as e:
        detections = []

    # Filter detections based on conf
    valid_detections = []
    for d in detections:
        c = float(d.get('confidence', 0))
        if c >= conf:
            valid_detections.append(d)

    # 7. Render Biometric Overlay & Extracted Face Crops
    annotated_b64, face_crops = create_biometric_overlay(pil_img, valid_detections)
    
    # 8. Check if Caleb is identified
    caleb_detected = any(c['is_caleb'] for c in face_crops)
    caleb_matches = [c for c in face_crops if c['is_caleb']]
    highest_conf = max([c['confidence_pct'] for c in face_crops], default=0.0)

    return jsonify({
        'success': True,
        'cloud_latency_ms': elapsed_ms,
        'total_faces': len(face_crops),
        'caleb_verified': caleb_detected,
        'target_name': TARGET_IDENTITY,
        'max_confidence': highest_conf,
        'image_dimensions': {'width': orig_w, 'height': orig_h},
        'original_image': orig_b64,
        'annotated_image': annotated_b64,
        'face_crops': face_crops,
        'raw_response': result_data
    })


if __name__ == '__main__':
    # Local dev server
    port = int(os.environ.get('PORT', 5000))
    print(f"==================================================")
    print(f" NeuraFace AI Engine — Caleb Biometric System")
    print(f" Palette: Option 5 (Bioluminescent Cyan & Deep Teal)")
    print(f" Endpoint: {DEFAULT_ENDPOINT}")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f"==================================================")
    app.run(host='0.0.0.0', port=port, debug=True)
