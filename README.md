# ArchaeoAI Backend - AI Archaeological Analysis API

🤖 Flask-based Backend for Archaeological Site Analysis & Computer Vision

A robust backend API service providing AI-powered computer vision capabilities for archaeological site analysis. Features include image segmentation, artifact detection, and statistical analysis of archaeological imagery with comprehensive REST API endpoints.

## 🎯 Core Capabilities
- **Image Processing**: Upload and manage archaeological site images
- **AI Segmentation**: Detect ruins, vegetation, and water bodies
- **Artifact Detection**: Identify and classify archaeological artifacts
- **Statistical Analysis**: Generate detailed site analysis reports
- **File Management**: Secure storage and retrieval of images/results

## 🏗️ Architecture
- **Framework**: Flask (Python)
- **Computer Vision**: OpenCV for image processing
- **AI/ML**: Custom algorithms for archaeological analysis
- **CORS**: Full CORS support for frontend integration
- **Deployment**: Render.com with Gunicorn WSGI

## 📊 API Endpoints
- GET / # Server status & endpoint list
- GET /api/health # Health check & system status
- GET /api/test # Connection test endpoint
- POST /api/real/upload # Upload archaeological images
- POST /api/real/segment # Segment site features
- POST /api/real/detect # Detect artifacts in images
- GET /uploads/<file> # Retrieve uploaded images
- GET /results/<file> # Retrieve processed results


## 🔧 Technology Stack
- **Backend**: Flask, Flask-CORS
- **Image Processing**: OpenCV, NumPy
- **File Handling**: Werkzeug secure uploads
- **Production**: Gunicorn WSGI server
- **Deployment**: Render.com cloud platform

## 📁 Project Structure
- ├── app.py               # Main Flask application
- ├── requirements.txt     # Python dependencies
- ├── uploads/            # User-uploaded images
- ├── results/            # Processed images & results
- ├── processed/          # Intermediate processed files
- └── README.md           # API documentation

## 🛡️ Features
- **Secure File Uploads**: Validation & sanitization
- **Error Handling**: Comprehensive error responses
- **CORS Configuration**: Full cross-origin support
- **Logging**: Detailed request/processing logs
- **Scalable**: Ready for production deployment

## 🌐 Deployment
- **Platform**: Render.com (Free Tier)
- **URL**: https://archaeological-backend.onrender.com
- **Status**: Auto-deploy from GitHub
- **Monitoring**: Built-in health checks

## 🎯 Use Cases
- Archaeological research platforms
- University research projects
- Cultural heritage documentation
- Field archaeology tools
- Educational applications

API Documentation available at: https://archaeological-backend.onrender.com/
