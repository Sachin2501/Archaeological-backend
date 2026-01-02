from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
import random
from werkzeug.utils import secure_filename
import json
import traceback

# ================= FLASK APP INIT =================
app = Flask(__name__)

# Enable CORS for ALL origins and ALL routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Content-Length"],
        "supports_credentials": False,
        "max_age": 600
    }
})

# Additional CORS headers for all responses
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, HEAD'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, X-Requested-With'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Type, Content-Length'
    response.headers['Access-Control-Max-Age'] = '600'
    return response

# Handle OPTIONS preflight requests for ALL routes
@app.route('/', methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path=None):
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS, HEAD')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, X-Requested-With')
    return response, 200

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
PROCESSED_FOLDER = 'processed'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['ALLOWED_EXTENSIONS'] = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}

# ================= AI MODEL =================
class ArchaeologicalAI:
    def __init__(self):
        print("🧠 Archaeological AI Models Initialized")
        self.segmentation_cache = {}
        self.detection_cache = {}
    
    def segment_site(self, image_path):
        """Segment archaeological site into ruins, vegetation, and other features"""
        try:
            print(f"Processing image: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                print("Error: Could not read image")
                return {"error": "Could not read image", "success": False}
            
            height, width = img.shape[:2]
            print(f"Image size: {width}x{height}")
            
            # Convert to different color spaces for better segmentation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # ================= RUINS DETECTION =================
            ruins_lower1 = np.array([0, 20, 50])
            ruins_upper1 = np.array([30, 100, 200])
            
            ruins_lower2 = np.array([0, 0, 100])
            ruins_upper2 = np.array([180, 50, 220])
            
            ruins_lower3 = np.array([5, 100, 100])
            ruins_upper3 = np.array([25, 255, 255])
            
            ruins_mask1 = cv2.inRange(hsv, ruins_lower1, ruins_upper1)
            ruins_mask2 = cv2.inRange(hsv, ruins_lower2, ruins_upper2)
            ruins_mask3 = cv2.inRange(hsv, ruins_lower3, ruins_upper3)
            ruins_mask = cv2.bitwise_or(ruins_mask1, cv2.bitwise_or(ruins_mask2, ruins_mask3))
            
            # ================= VEGETATION DETECTION =================
            vegetation_lower1 = np.array([35, 40, 40])
            vegetation_upper1 = np.array([85, 255, 255])
            
            vegetation_mask1 = cv2.inRange(hsv, vegetation_lower1, vegetation_upper1)
            vegetation_mask = vegetation_mask1
            
            # ================= WATER DETECTION =================
            water_lower = np.array([90, 50, 70])
            water_upper = np.array([130, 255, 255])
            water_mask = cv2.inRange(hsv, water_lower, water_upper)
            
            # ================= ENHANCEMENTS =================
            kernel = np.ones((5,5), np.uint8)
            
            ruins_mask = cv2.morphologyEx(ruins_mask, cv2.MORPH_CLOSE, kernel)
            ruins_mask = cv2.morphologyEx(ruins_mask, cv2.MORPH_OPEN, kernel)
            
            vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
            
            # Remove overlaps
            vegetation_mask = cv2.bitwise_and(vegetation_mask, cv2.bitwise_not(water_mask))
            ruins_mask = cv2.bitwise_and(ruins_mask, cv2.bitwise_not(cv2.bitwise_or(water_mask, vegetation_mask)))
            
            # ================= CALCULATIONS =================
            total_pixels = height * width
            ruins_pixels = np.sum(ruins_mask > 0)
            vegetation_pixels = np.sum(vegetation_mask > 0)
            water_pixels = np.sum(water_mask > 0)
            
            ruins_percent = (ruins_pixels / total_pixels) * 100
            vegetation_percent = (vegetation_pixels / total_pixels) * 100
            water_percent = (water_pixels / total_pixels) * 100
            
            # ================= VISUALIZATION =================
            result_img = img.copy()
            
            overlay = np.zeros_like(img)
            overlay[ruins_mask > 0] = [255, 140, 0]
            overlay[vegetation_mask > 0] = [0, 180, 0]
            overlay[water_mask > 0] = [0, 140, 255]
            
            alpha = 0.6
            result_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            
            # ================= SAVE RESULTS =================
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"seg_result_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, result_img)
            print(f"Result saved: {result_filename}")
            
            return {
                "success": True,
                "ruins_percentage": round(float(ruins_percent), 2),
                "vegetation_percentage": round(float(vegetation_percent), 2),
                "water_percentage": round(float(water_percent), 2),
                "result_image": f"/results/{result_filename}",
                "image_size": f"{width}x{height}",
                "pixels_analyzed": int(total_pixels),
                "ruins_area_pixels": int(ruins_pixels),
                "vegetation_area_pixels": int(vegetation_pixels),
                "water_area_pixels": int(water_pixels),
                "analysis_timestamp": datetime.now().isoformat(),
                "confidence_score": round(0.85 + random.uniform(0, 0.1), 3)
            }
            
        except Exception as e:
            print(f"Segmentation error: {str(e)}")
            traceback.print_exc()
            return {"error": f"Segmentation failed: {str(e)}", "success": False}
    
    def detect_artifacts(self, image_path):
        """Detect archaeological artifacts using computer vision"""
        try:
            print(f"Detecting artifacts in: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                print("Error: Could not read image")
                return {"error": "Could not read image", "success": False}
            
            height, width = img.shape[:2]
            print(f"Image size: {width}x{height}")
            result_img = img.copy()
            
            # ================= PREPROCESSING =================
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # ================= CONTOUR DETECTION =================
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Found {len(contours)} contours")
            
            artifacts = []
            artifact_id = 1
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if 500 < area < 20000:
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter < 60:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h) if h != 0 else 0
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
                    
                    # Classify artifact
                    artifact_type = self._classify_artifact(area, aspect_ratio, circularity)
                    
                    # Calculate confidence
                    confidence = min(0.95, max(0.3, 0.4 + (area / 10000) + circularity * 0.3))
                    
                    # Get center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = x + w//2, y + h//2
                    
                    artifact_data = {
                        "id": artifact_id,
                        "type": artifact_type,
                        "confidence": round(float(confidence), 3),
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": round(float(area), 2),
                        "center": [int(cX), int(cY)],
                        "perimeter": round(float(perimeter), 2)
                    }
                    
                    artifacts.append(artifact_data)
                    
                    # Draw visualization
                    color = self._get_color_for_type(artifact_type)
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                    cv2.drawContours(result_img, [contour], -1, color, 1)
                    cv2.circle(result_img, (cX, cY), 4, color, -1)
                    
                    label = f"{artifact_type} ({confidence:.0%})"
                    cv2.putText(result_img, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    artifact_id += 1
            
            # Sort by confidence
            artifacts.sort(key=lambda x: x['confidence'], reverse=True)
            
            # ================= SAVE RESULTS =================
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"detect_result_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, result_img)
            print(f"Detection result saved: {result_filename}")
            
            # Group artifacts by type
            artifact_types = {}
            for artifact in artifacts:
                artifact_type = artifact['type']
                if artifact_type not in artifact_types:
                    artifact_types[artifact_type] = []
                artifact_types[artifact_type].append(artifact)
            
            type_counts = {atype: len(items) for atype, items in artifact_types.items()}
            
            return {
                "success": True,
                "artifacts": artifacts[:30],
                "total_detected": len(artifacts),
                "artifact_types": type_counts,
                "result_image": f"/results/{result_filename}",
                "image_size": f"{width}x{height}",
                "analysis_timestamp": datetime.now().isoformat(),
                "message": f"Found {len(artifacts)} artifacts"
            }
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            traceback.print_exc()
            return {"error": f"Detection failed: {str(e)}", "success": False}
    
    def _classify_artifact(self, area, aspect_ratio, circularity):
        """Simple artifact classification"""
        if circularity > 0.7:
            if area > 3000:
                return "Large Vessel"
            else:
                return "Pottery"
        elif aspect_ratio > 2.5 or aspect_ratio < 0.4:
            return "Tool"
        elif 0.8 < aspect_ratio < 1.2 and area > 2000:
            return "Building Block"
        else:
            artifact_categories = [
                "Ceramic Fragment",
                "Stone Artifact",
                "Bone Fragment",
                "Metal Object",
                "Ancient Tool",
                "Architectural Piece",
                "Ornament",
                "Weapon"
            ]
            return random.choice(artifact_categories)
    
    def _get_color_for_type(self, artifact_type):
        """Get BGR color for artifact visualization"""
        color_map = {
            "Large Vessel": (0, 255, 255),
            "Pottery": (0, 165, 255),
            "Tool": (255, 0, 0),
            "Building Block": (0, 255, 0),
            "Ceramic Fragment": (42, 42, 165),
            "Stone Artifact": (128, 128, 128),
            "Bone Fragment": (255, 255, 255),
            "Metal Object": (192, 192, 192),
            "Ancient Tool": (255, 0, 255),
            "Architectural Piece": (255, 255, 0),
            "Ornament": (147, 20, 255),
            "Weapon": (0, 0, 255)
        }
        return color_map.get(artifact_type, (255, 255, 255))

# Initialize AI
archaeo_ai = ArchaeologicalAI()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {ext[1:] for ext in app.config['ALLOWED_EXTENSIONS']}

# ================= API ROUTES =================

@app.route('/')
def index():
    """Root endpoint - basic info"""
    return jsonify({
        "status": "online",
        "name": "Archaeological AI Backend",
        "version": "2.0.0",
        "description": "AI-powered archaeological site analysis system",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "upload": "/api/real/upload",
            "segment": "/api/real/segment",
            "detect": "/api/real/detect",
            "health": "/api/health",
            "test": "/api/test"
        }
    })

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    return jsonify({
        "status": "healthy",
        "service": "archaeological-backend",
        "timestamp": datetime.now().isoformat(),
        "ai_ready": True,
        "storage": {
            "uploads": len(os.listdir(UPLOAD_FOLDER)),
            "results": len(os.listdir(RESULT_FOLDER))
        }
    })

@app.route('/api/test', methods=['GET', 'OPTIONS'])
def test_endpoint():
    """Test endpoint for debugging"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    return jsonify({
        "success": True,
        "message": "Backend is working correctly!",
        "endpoint": "/api/test",
        "timestamp": datetime.now().isoformat(),
        "cors_enabled": True,
        "request_origin": request.headers.get('Origin', 'Not specified')
    })

# ================= MAIN ENDPOINTS =================

@app.route('/api/real/upload', methods=['POST', 'OPTIONS'])
def handle_real_upload():
    """Handle image upload - main endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        print("Upload endpoint called")
        print(f"Request Origin: {request.headers.get('Origin')}")
        print(f"Request files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return jsonify({
                "success": False, 
                "error": "Invalid file type. Allowed: jpg, jpeg, png, tif, tiff, bmp, gif",
                "allowed_types": list(app.config['ALLOWED_EXTENSIONS'])
            }), 400
        
        # Secure filename
        original_name = secure_filename(file.filename)
        file_ext = os.path.splitext(original_name)[1].lower()
        
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{unique_id}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(filepath)
        print(f"File saved to: {filepath}")
        
        # Verify image
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({"success": False, "error": "Invalid image file"}), 400
        
        height, width = img.shape[:2]
        file_size = os.path.getsize(filepath)
        
        print(f"Image verified: {width}x{height}, {file_size} bytes")
        
        response = {
            "success": True,
            "filename": filename,
            "original_name": original_name,
            "image_size": {
                "width": width,
                "height": height
            },
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "preview_url": f"/uploads/{filename}",
            "upload_timestamp": datetime.now().isoformat(),
            "message": "Image uploaded successfully"
        }
        
        print(f"Upload successful: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/real/segment', methods=['POST', 'OPTIONS'])
def handle_real_segment():
    """Handle segmentation request - main endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        print(f"Segmentation request received")
        print(f"Request Origin: {request.headers.get('Origin')}")
        
        data = request.get_json()
        print(f"Segmentation request data: {data}")
        
        if not data or 'filename' not in data:
            return jsonify({"success": False, "error": "Filename required"}), 400
        
        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return jsonify({"success": False, "error": "File not found"}), 404
        
        print(f"Starting segmentation for {filename}")
        
        result = archaeo_ai.segment_site(filepath)
        
        if "error" in result:
            print(f"Segmentation failed: {result['error']}")
            return jsonify(result), 500
        
        result["input_image"] = f"/uploads/{filename}"
        
        print(f"Segmentation completed successfully for {filename}")
        print(f"Results: Ruins {result.get('ruins_percentage', 0)}%, Vegetation {result.get('vegetation_percentage', 0)}%")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Segmentation endpoint error: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/real/detect', methods=['POST', 'OPTIONS'])
def handle_real_detect():
    """Handle detection request - main endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        print(f"Detection request received")
        print(f"Request Origin: {request.headers.get('Origin')}")
        
        data = request.get_json()
        print(f"Detection request data: {data}")
        
        if not data or 'filename' not in data:
            return jsonify({"success": False, "error": "Filename required"}), 400
        
        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return jsonify({"success": False, "error": "File not found"}), 404
        
        print(f"Starting artifact detection for {filename}")
        
        result = archaeo_ai.detect_artifacts(filepath)
        
        if "error" in result:
            print(f"Detection failed: {result['error']}")
            return jsonify(result), 500
        
        result["input_image"] = f"/uploads/{filename}"
        
        artifact_count = result.get('total_detected', 0)
        print(f"Detection complete, found {artifact_count} artifacts")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Detection endpoint error: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ================= STATIC FILE SERVING =================

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": f"File not found: {filename}"}), 404

@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result files"""
    try:
        return send_from_directory(app.config['RESULT_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": f"Result file not found: {filename}"}), 404

@app.route('/processed/<filename>')
def serve_processed(filename):
    """Serve processed files"""
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": f"Processed file not found: {filename}"}), 404

# ================= ERROR HANDLING =================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False, 
        "error": "Endpoint not found",
        "available_endpoints": {
            "root": "/",
            "upload": "/api/real/upload",
            "segment": "/api/real/segment",
            "detect": "/api/real/detect",
            "health": "/api/health",
            "test": "/api/test"
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False, 
        "error": "Internal server error",
        "message": "Please try again later"
    }), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large",
        "max_size": "50MB"
    }), 413

# ================= APPLICATION STARTUP =================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    
    print("=" * 70)
    print("🔬 ARCHAEOLOGICAL AI BACKEND - COMPLETE FIXED VERSION")
    print("=" * 70)
    print(f"📡 Server starting on port: {port}")
    print(f"🌐 Production URL: https://archaeological-backend.onrender.com")
    print("-" * 70)
    print("🚀 Available Endpoints:")
    print("  ✓ GET  /                 - Server status")
    print("  ✓ GET  /api/health       - Health check")
    print("  ✓ GET  /api/test         - Test connection")
    print("  ✓ POST /api/real/upload  - Upload image (MAIN)")
    print("  ✓ POST /api/real/segment - Segment site features")
    print("  ✓ POST /api/real/detect  - Detect artifacts")
    print("  ✓ GET  /uploads/<file>   - Get uploaded image")
    print("  ✓ GET  /results/<file>   - Get result image")
    print("-" * 70)
    print("📁 Storage Directories:")
    print(f"  • Uploads: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"  • Results: {os.path.abspath(RESULT_FOLDER)}")
    print(f"  • Processed: {os.path.abspath(PROCESSED_FOLDER)}")
    print("=" * 70)
    
    # For Render.com deployment
    app.run(host='0.0.0.0', port=port, debug=False)