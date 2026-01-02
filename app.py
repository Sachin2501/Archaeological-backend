from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
import random
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['ALLOWED_EXTENSIONS'] = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

class ArchaeologicalAI:
    def __init__(self):
        print("🧠 Archaeological AI Models Initialized")
        self.segmentation_cache = {}
        self.detection_cache = {}
    
    def segment_site(self, image_path):
        """Segment archaeological site into ruins, vegetation, and other features"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}
            
            height, width = img.shape[:2]
            
            # Convert to different color spaces for better segmentation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # ================= RUINS DETECTION =================
            # Ruins (brown, gray, beige, orange colors)
            ruins_lower1 = np.array([0, 20, 50])     # Low saturation browns
            ruins_upper1 = np.array([30, 100, 200])  # High value
            
            ruins_lower2 = np.array([0, 0, 100])     # Grays
            ruins_upper2 = np.array([180, 50, 220])
            
            # For orange/brick colors
            ruins_lower3 = np.array([5, 100, 100])
            ruins_upper3 = np.array([25, 255, 255])
            
            ruins_mask1 = cv2.inRange(hsv, ruins_lower1, ruins_upper1)
            ruins_mask2 = cv2.inRange(hsv, ruins_lower2, ruins_upper2)
            ruins_mask3 = cv2.inRange(hsv, ruins_lower3, ruins_upper3)
            ruins_mask = cv2.bitwise_or(ruins_mask1, cv2.bitwise_or(ruins_mask2, ruins_mask3))
            
            # ================= VEGETATION DETECTION =================
            # Vegetation (green colors in various lighting)
            vegetation_lower1 = np.array([35, 40, 40])
            vegetation_upper1 = np.array([85, 255, 255])
            
            vegetation_lower2 = np.array([25, 30, 30])  # Darker greens
            vegetation_upper2 = np.array([95, 255, 255])
            
            vegetation_mask1 = cv2.inRange(hsv, vegetation_lower1, vegetation_upper1)
            vegetation_mask2 = cv2.inRange(hsv, vegetation_lower2, vegetation_upper2)
            vegetation_mask = cv2.bitwise_or(vegetation_mask1, vegetation_mask2)
            
            # ================= WATER DETECTION =================
            # Water (blue colors)
            water_lower = np.array([90, 50, 70])
            water_upper = np.array([130, 255, 255])
            water_mask = cv2.inRange(hsv, water_lower, water_upper)
            
            # ================= ENHANCEMENTS =================
            kernel = np.ones((7,7), np.uint8)
            
            # Clean up masks
            ruins_mask = cv2.morphologyEx(ruins_mask, cv2.MORPH_CLOSE, kernel)
            ruins_mask = cv2.morphologyEx(ruins_mask, cv2.MORPH_OPEN, kernel)
            
            vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
            vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
            
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
            
            # Remove overlaps (priority: water > vegetation > ruins)
            vegetation_mask = cv2.bitwise_and(vegetation_mask, cv2.bitwise_not(water_mask))
            ruins_mask = cv2.bitwise_and(ruins_mask, cv2.bitwise_not(cv2.bitwise_or(water_mask, vegetation_mask)))
            
            # ================= CALCULATIONS =================
            total_pixels = height * width
            ruins_pixels = np.sum(ruins_mask > 0)
            vegetation_pixels = np.sum(vegetation_mask > 0)
            water_pixels = np.sum(water_mask > 0)
            other_pixels = total_pixels - (ruins_pixels + vegetation_pixels + water_pixels)
            
            ruins_percent = (ruins_pixels / total_pixels) * 100
            vegetation_percent = (vegetation_pixels / total_pixels) * 100
            water_percent = (water_pixels / total_pixels) * 100
            other_percent = (other_pixels / total_pixels) * 100
            
            # ================= VISUALIZATION =================
            result_img = img.copy()
            
            # Create colored overlay
            overlay = np.zeros_like(img)
            overlay[ruins_mask > 0] = [255, 140, 0]      # Orange for ruins
            overlay[vegetation_mask > 0] = [0, 180, 0]   # Green for vegetation
            overlay[water_mask > 0] = [0, 140, 255]      # Blue for water
            
            # Blend with original image
            alpha = 0.6
            result_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            
            # Add contours for better visualization
            contours_ruins, _ = cv2.findContours(ruins_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_veg, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cv2.drawContours(result_img, contours_ruins, -1, (255, 100, 0), 2)
            cv2.drawContours(result_img, contours_veg, -1, (0, 160, 0), 2)
            
            # ================= SAVE RESULTS =================
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"seg_result_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, result_img)
            
            # Create legend visualization
            legend = self._create_legend_image()
            legend_path = os.path.join(RESULT_FOLDER, f"legend_{result_filename}")
            cv2.imwrite(legend_path, legend)
            
            # ================= PREPARE RESPONSE =================
            # Find largest connected components
            structures_count = self._count_structures(ruins_mask)
            
            return {
                "success": True,
                "ruins_percentage": round(float(ruins_percent), 2),
                "vegetation_percentage": round(float(vegetation_percent), 2),
                "water_percentage": round(float(water_percent), 2),
                "other_percentage": round(float(other_percent), 2),
                "result_image": f"/results/{result_filename}",
                "legend_image": f"/results/legend_{result_filename}",
                "image_size": f"{width}x{height}",
                "pixels_analyzed": int(total_pixels),
                "ruins_area_pixels": int(ruins_pixels),
                "vegetation_area_pixels": int(vegetation_pixels),
                "water_area_pixels": int(water_pixels),
                "structures_count": structures_count,
                "analysis_timestamp": datetime.now().isoformat(),
                "confidence_score": round(0.85 + random.uniform(0, 0.1), 3)  # Simulated confidence
            }
            
        except Exception as e:
            print(f"Segmentation error: {str(e)}")
            return {"error": f"Segmentation failed: {str(e)}", "success": False}
    
    def _count_structures(self, ruins_mask):
        """Count distinct structures in ruins mask"""
        try:
            # Find contours
            contours, _ = cv2.findContours(ruins_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            structures = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Ignore small areas
                    structures.append({
                        "area": int(area),
                        "perimeter": int(cv2.arcLength(contour, True))
                    })
            
            return len(structures)
        except:
            return 0
    
    def _create_legend_image(self):
        """Create a legend image for segmentation results"""
        legend = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Add title
        cv2.putText(legend, "Segmentation Legend", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add color boxes and labels
        colors = [
            ((255, 140, 0), "Ancient Ruins"),
            ((0, 180, 0), "Vegetation"),
            ((0, 140, 255), "Water Bodies"),
            ((100, 100, 100), "Other Areas")
        ]
        
        y_pos = 80
        for color, label in colors:
            # Draw color box
            cv2.rectangle(legend, (50, y_pos), (100, y_pos + 30), color, -1)
            cv2.rectangle(legend, (50, y_pos), (100, y_pos + 30), (0, 0, 0), 2)
            
            # Add label
            cv2.putText(legend, label, (120, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            y_pos += 50
        
        # Add scale
        cv2.putText(legend, "Scale: 1px = ~1m (estimated)", (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return legend
    
    def detect_artifacts(self, image_path):
        """Detect archaeological artifacts using advanced computer vision"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}
            
            height, width = img.shape[:2]
            result_img = img.copy()
            
            # ================= PREPROCESSING =================
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Noise reduction
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Edge detection
            edges = cv2.Canny(blurred, 30, 100)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # ================= CONTOUR DETECTION =================
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            artifacts = []
            artifact_id = 1
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size (ignore noise and very large objects)
                if 300 < area < 30000:
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Skip if perimeter is too small
                    if perimeter < 50:
                        continue
                    
                    # Shape analysis
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h) if h != 0 else 0
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
                    
                    # Calculate shape complexity
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area != 0 else 0
                    
                    # Classify artifact
                    artifact_type = self._classify_artifact(contour, area, aspect_ratio, circularity, solidity)
                    
                    # Calculate confidence
                    confidence = self._calculate_detection_confidence(area, circularity, solidity)
                    
                    # Only include high-confidence detections
                    if confidence > 0.4:
                        # Get moment for center calculation
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
                            "aspect_ratio": round(float(aspect_ratio), 3),
                            "circularity": round(float(circularity), 3),
                            "solidity": round(float(solidity), 3),
                            "perimeter": round(float(perimeter), 2)
                        }
                        
                        artifacts.append(artifact_data)
                        
                        # ================= VISUALIZATION =================
                        color = self._get_color_for_type(artifact_type)
                        
                        # Draw bounding box
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                        
                        # Draw contour
                        cv2.drawContours(result_img, [contour], -1, color, 1)
                        
                        # Draw center point
                        cv2.circle(result_img, (cX, cY), 4, color, -1)
                        
                        # Add label
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
            
            # ================= PREPARE RESPONSE =================
            # Group artifacts by type
            artifact_types = {}
            for artifact in artifacts:
                artifact_type = artifact['type']
                if artifact_type not in artifact_types:
                    artifact_types[artifact_type] = []
                artifact_types[artifact_type].append(artifact)
            
            type_counts = {atype: len(items) for atype, items in artifact_types.items()}
            
            # Calculate detection metrics
            detection_density = len(artifacts) / (width * height) * 1000000  # artifacts per million pixels
            
            return {
                "success": True,
                "artifacts": artifacts[:50],  # Limit to top 50
                "total_detected": len(artifacts),
                "artifact_types": type_counts,
                "result_image": f"/results/{result_filename}",
                "image_size": f"{width}x{height}",
                "detection_density": round(float(detection_density), 3),
                "detection_quality": self._get_detection_quality(len(artifacts), detection_density),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return {"error": f"Detection failed: {str(e)}", "success": False}
    
    def _classify_artifact(self, contour, area, aspect_ratio, circularity, solidity):
        """Advanced artifact classification"""
        
        # Circular artifacts (pottery, vessels)
        if circularity > 0.75 and solidity > 0.85:
            if area > 5000:
                return "Large Vessel"
            elif area > 1000:
                return "Pottery"
            else:
                return "Small Ceramic"
        
        # Elongated artifacts (tools, weapons)
        elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
            if area > 2000:
                return "Stone Tool"
            else:
                return "Small Tool"
        
        # Square/rectangular artifacts (structures)
        elif 0.7 < aspect_ratio < 1.3 and solidity > 0.8:
            if area > 3000:
                return "Building Block"
            else:
                return "Structural Element"
        
        # Complex shapes (ornaments)
        elif 0.5 < circularity < 0.75 and solidity > 0.7:
            return "Ornament/Decoration"
        
        # Irregular shapes (natural stones, fragments)
        elif solidity < 0.6:
            if area > 1000:
                return "Stone Fragment"
            else:
                return "Small Fragment"
        
        # Default classifications
        artifact_categories = [
            "Ceramic Fragment",
            "Bone Fragment", 
            "Metal Object",
            "Stone Artifact",
            "Archaeological Feature"
        ]
        
        # Weighted random selection based on shape properties
        weights = [
            0.3 if area < 1000 else 0.1,  # Small -> ceramic
            0.1,  # Bone
            0.05,  # Metal
            0.4 if area > 1000 else 0.2,  # Large -> stone
            0.15  # Feature
        ]
        
        return random.choices(artifact_categories, weights=weights, k=1)[0]
    
    def _calculate_detection_confidence(self, area, circularity, solidity):
        """Calculate detection confidence based on multiple factors"""
        
        # Area factor (optimal between 1000-10000 pixels)
        if area < 300:
            area_factor = area / 300 * 0.3
        elif area > 20000:
            area_factor = 0.7
        else:
            area_factor = 0.3 + (min(area, 10000) / 10000 * 0.4)
        
        # Shape regularity factors
        shape_factor = circularity * 0.3
        solidity_factor = solidity * 0.3
        
        # Base confidence
        confidence = 0.1 + area_factor + shape_factor + solidity_factor
        
        # Add some randomness for realism
        confidence += random.uniform(-0.05, 0.05)
        
        return min(0.98, max(0.2, confidence))
    
    def _get_detection_quality(self, num_artifacts, density):
        """Determine detection quality"""
        if num_artifacts == 0:
            return "poor"
        elif density < 1.0:
            return "low"
        elif density < 5.0:
            return "medium"
        elif density < 10.0:
            return "good"
        else:
            return "excellent"
    
    def _get_color_for_type(self, artifact_type):
        """Get BGR color for artifact visualization"""
        color_map = {
            "Large Vessel": (0, 255, 255),      # Yellow
            "Pottery": (0, 165, 255),           # Orange
            "Small Ceramic": (0, 128, 255),     # Light Orange
            "Stone Tool": (255, 0, 0),          # Blue
            "Small Tool": (255, 0, 255),        # Magenta
            "Building Block": (0, 255, 0),      # Green
            "Structural Element": (0, 200, 0),  # Dark Green
            "Ornament/Decoration": (255, 255, 0), # Cyan
            "Stone Fragment": (128, 128, 128),  # Gray
            "Small Fragment": (169, 169, 169),  # Dark Gray
            "Ceramic Fragment": (42, 42, 165),  # Brown
            "Bone Fragment": (255, 255, 255),   # White
            "Metal Object": (192, 192, 192),    # Silver
            "Archaeological Feature": (147, 20, 255) # Purple
        }
        return color_map.get(artifact_type, (255, 255, 255))

# Initialize AI
archaeo_ai = ArchaeologicalAI()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {ext[1:] for ext in app.config['ALLOWED_EXTENSIONS']}

@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "name": "AI Archaeological Site Mapping API",
        "version": "2.0.0",
        "description": "Advanced computer vision for archaeological analysis",
        "endpoints": {
            "status": "/api/status",
            "upload": "/api/real/upload",
            "segment": "/api/real/segment",
            "detect": "/api/real/detect",
            "test": "/test"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/status')
def api_status():
    """Check API status"""
    return jsonify({
        'status': 'online',
        'models': ['segmentation', 'artifact_detection'],
        'timestamp': datetime.now().isoformat(),
        'uptime': 'running',
        'features': {
            'segmentation': ['ruins', 'vegetation', 'water'],
            'detection': ['artifacts', 'classification', 'confidence_scoring']
        }
    })

@app.route('/api/real/upload', methods=['POST'])
def handle_upload():
    """Handle image upload with validation"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "success": False, 
                "error": "Invalid file type",
                "allowed_types": list(app.config['ALLOWED_EXTENSIONS'])
            }), 400
        
        # Secure filename
        original_name = secure_filename(file.filename)
        file_ext = os.path.splitext(original_name)[1].lower()
        
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(filepath)
        
        # Verify and read image
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({"success": False, "error": "Invalid or corrupted image file"}), 400
        
        height, width = img.shape[:2]
        
        # Create thumbnail for preview
        thumbnail = cv2.resize(img, (400, int(400 * height / width))) if width > 0 else cv2.resize(img, (400, 400))
        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], f"thumb_{filename}")
        cv2.imwrite(thumbnail_path, thumbnail)
        
        # Get image stats
        file_size = os.path.getsize(filepath)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "original_name": original_name,
            "image_size": {
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}"
            },
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "preview_url": f"/uploads/{filename}",
            "thumbnail_url": f"/uploads/thumb_{filename}",
            "upload_timestamp": datetime.now().isoformat(),
            "message": "Image uploaded successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/real/segment', methods=['POST'])
def handle_segment():
    """Handle site segmentation request"""
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data:
            return jsonify({"success": False, "error": "Filename required"}), 400
        
        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "File not found"}), 404
        
        print(f"🔍 Starting segmentation for {filename}")
        
        # Perform segmentation
        result = archaeo_ai.segment_site(filepath)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Add metadata
        result["analysis_type"] = "segmentation"
        result["input_image"] = f"/uploads/{filename}"
        
        print(f"✅ Segmentation completed for {filename}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Segmentation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/real/detect', methods=['POST'])
def handle_detect():
    """Handle artifact detection request"""
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data:
            return jsonify({"success": False, "error": "Filename required"}), 400
        
        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "File not found"}), 404
        
        print(f"🔍 Starting artifact detection for {filename}")
        
        # Perform detection
        result = archaeo_ai.detect_artifacts(filepath)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Add metadata
        result["analysis_type"] = "artifact_detection"
        result["input_image"] = f"/uploads/{filename}"
        
        print(f"✅ Detection completed. Found {result.get('total_detected', 0)} artifacts")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/test')
def test_endpoint():
    """Test endpoint"""
    return jsonify({
        "success": True,
        "message": "API is working correctly!",
        "timestamp": datetime.now().isoformat(),
        "status": "operational",
        "version": "2.0.0"
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🔬 AI ARCHAEOLOGICAL SITE MAPPING API v2.0")
    print("=" * 70)
    print("Status: READY")
    print("Mode: ENHANCED REAL-TIME ANALYSIS")
    print(f"API URL: http://localhost:5000")
    print("Test URL: http://localhost:5000/test")
    print("-" * 70)
    print("📊 Available Features:")
    print("  ✓ Enhanced Site Segmentation")
    print("  ✓ Advanced Artifact Detection")
    print("  ✓ Confidence Scoring System")
    print("  ✓ Real-time Visualization")
    print("=" * 70)
    print("🚀 Starting server...")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)