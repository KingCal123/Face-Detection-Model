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
    Face area remains 100% visible and un-obscured (NO solid fills over faces).
    Returns: (annotated_base64_data_url, list_of_face_crops_with_metadata)
    """
    annotated = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    
    # Adaptive scalable font
    font_size = max(13, int(min(w, h) * 0.026))
    try:
        font = ImageFont.load_default(size=font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        
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
        primary_color = (6, 182, 212) if is_target else (45, 212, 191)
        accent_color = (34, 211, 238)
        
        # 1. Thin bounding box perimeter (NO inner fill so face is 100% natural & visible)
        draw.rectangle([x1, y1, x2, y2], fill=None, outline=primary_color, width=1)
        
        # 2. Biometric Corner Reticles (tactical L-brackets on 4 corners)
        corner_len = max(10, int(min(bw, bh) * 0.18))
        corner_width = max(2, int(min(w, h) * 0.004))
        
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
        
        # 3. Center Micro-Crosshair
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ch_size = max(4, int(min(bw, bh) * 0.05))
        draw.line([(cx - ch_size, cy), (cx + ch_size, cy)], fill=accent_color, width=1)
        draw.line([(cx, cy - ch_size), (cx, cy + ch_size)], fill=accent_color, width=1)
        
        # 4. Compact, proportional Badge that fits the text snugly without obstructing
        label_text = f"● {name.upper()} · {conf * 100:.1f}%"
        try:
            tb = draw.textbbox((0, 0), label_text, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            offset_y = tb[1]
        except Exception:
            tw, th = len(label_text) * 8, 14
            offset_y = 0
            
        pad_x = 7
        pad_y = 4
        badge_w = tw + (pad_x * 2)
        badge_h = th + (pad_y * 2)
        
        # Place neatly above box if space permits, else inside top edge
        if y1 >= badge_h + 6:
            bx1 = x1
            by1 = y1 - badge_h - 4
        else:
            bx1 = x1 + 4
            by1 = y1 + 4
            
        bx2 = bx1 + badge_w
        by2 = by1 + badge_h
        
        draw.rectangle([bx1, by1, bx2, by2], fill=(6, 16, 23), outline=primary_color, width=1)
        draw.text((bx1 + pad_x, by1 + pad_y - offset_y), label_text, fill=accent_color, font=font)
        
        # 5. Extract Face Crop with proportional padding for the gallery
        pad_x_crop = bw * 0.15
        pad_y_crop = bh * 0.20
        crop_x1 = max(0, int(x1 - pad_x_crop))
        crop_y1 = max(0, int(y1 - pad_y_crop))
        crop_x2 = min(w, int(x2 + pad_x_crop))
        crop_y2 = min(h, int(y2 + pad_y_crop))
        
        face_crop = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        # Resize thumbnail to square 200x200
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
        with urllib.request.urlopen(req, timeout=12) as resp:
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
    port = int(os.environ.get('PORT', 5000))
    print(f"==================================================")
    print(f" NeuraFace AI Engine — Caleb Biometric System")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f"==================================================")
    app.run(host='0.0.0.0', port=port, debug=True)
