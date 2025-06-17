# 🎭 AI Face Swap Studio

**Advanced AI-powered face swapping application with real-time camera integration and user authentication.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

---

## 📋 **Table of Contents**

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔧 Configuration](#-configuration)
- [💻 Usage](#-usage)
- [📱 User Guide](#-user-guide)
- [🛠️ API Documentation](#️-api-documentation)
- [🏗️ Architecture](#️-architecture)
- [🎨 Frontend](#-frontend)
- [🔒 Security](#-security)
- [📊 Performance](#-performance)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ **Features**

### 🎯 **Core Features**
- **Real-time Face Swapping** with InsightFace AI technology
- **Auto Countdown Camera** with 5-second timer
- **User Authentication** with SQLite database
- **Clean URL Routing** (`/login`, `/character`, `/camera`, `/result`)
- **QR Code Generation** for easy mobile downloads
- **Mobile Responsive** design with touch optimization
- **Frame Overlay Effects** for enhanced results

### 🔐 **Security & Authentication**
- JWT token-based authentication
- Password hashing with PBKDF2 + salt
- File upload validation and sanitization
- User session management
- Secure API endpoints

### 📱 **User Experience**
- **Streamlined Workflow**: Login → Select → Capture → Download
- **Auto Processing**: No manual clicking required
- **Multiple Download Options**: QR code, direct download, share
- **Real-time Feedback**: Loading states and error handling
- **Mobile Optimized**: Touch-friendly interface

### 🛠️ **Technical Features**
- **FastAPI Backend** with async operations
- **SQLite Database** with migration support
- **File Management** with automatic cleanup
- **Error Recovery** with graceful fallbacks
- **Comprehensive Logging** for debugging

---

## 🚀 **Quick Start**

### **1. Clone & Setup**
```bash
git clone https://github.com/apepsiii/faceswap.git
cd faceswap
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Download Face Swap Model**
```bash
# Download inswapper_128.onnx model (not included due to size)
# Place in project root directory
```

### **4. Run Application**
```bash
python main.py
```

### **5. Access Application**
- **URL**: http://localhost:5000
- **Demo Login**: username: `demo`, password: `demo123`

---

## 📦 **Installation**

### **System Requirements**
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: Webcam for photo capture
- **Browser**: Chrome, Firefox, Safari, Edge (latest versions)

### **Dependencies**
```bash
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# File Handling
python-multipart==0.0.6
aiofiles==23.2.1

# AI & Computer Vision
opencv-python==4.8.1.78
insightface==0.7.3
numpy==1.24.3
Pillow==10.0.1

# Authentication
PyJWT==2.8.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

## 🔧 **Configuration**

### **Environment Variables**
Create `.env` file (optional):
```env
# Server Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=True

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_EXPIRATION_HOURS=24

# File Upload
MAX_FILE_SIZE=10485760  # 10MB
UPLOAD_DIR=static/uploads
RESULT_DIR=static/results

# Face Detection
DET_SIZE=640,640
CTX_ID=0  # 0 for GPU, -1 for CPU
```

### **Application Settings**
Edit `main.py` configuration:
```python
class Config:
    UPLOAD_DIR = Path("static/uploads")
    TEMPLATE_DIR = Path("static/templates")
    RESULT_DIR = Path("static/results")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    DET_SIZE = (640, 640)
    CTX_ID = 0  # GPU acceleration
```

---

## 💻 **Usage**

### **Development Mode**
```bash
# Run with auto-reload
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### **Production Mode**
```bash
# Single worker
uvicorn main:app --host 0.0.0.0 --port 5000

# Multiple workers
uvicorn main:app --host 0.0.0.0 --port 5000 --workers 4

# With Gunicorn (Linux/Mac)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
```

## 📱 **User Guide**

### **Step-by-Step Usage**

#### **1. Login/Register**
- Visit http://localhost:5000
- Use demo account: `faceswapp` / `demo123`
- Or register new account with username/password

#### **2. Select Character Template**
- Choose from available character templates
- Templates include: Superhero, Princess, Warrior, Cartoon
- Click template to select, then "Continue to Camera"

#### **3. Auto Photo Capture**
- Position face within the guide circle
- Click "Ready Photo" button
- Wait for 5-second countdown
- Photo automatically captured and processed

#### **4. Download Results**
- View generated face swap result
- Scan QR code with phone for instant download
- Or use direct download button
- Share with friends using share button

### **User Features**

#### **Camera Controls**
- **Switch Camera**: Toggle between front/back camera
- **Auto Countdown**: 5-second timer with visual feedback
- **Face Guide**: Visual positioning help
- **Auto Processing**: No manual intervention required

#### **Download Options**
- **QR Code**: Scan with phone camera
- **Direct Download**: Click download button
- **Share Function**: Copy link to clipboard
- **Manual Link**: Backup download option

---

## 🛠️ **API Documentation**

### **Authentication Endpoints**

#### **POST /api/register**
Register new user account.
```json
{
  "username": "string",
  "password": "string"
}
```

#### **POST /api/login**
Authenticate user and get JWT token.
```json
{
  "username": "string", 
  "password": "string"
}
```

#### **GET /api/me**
Get current user information (requires authentication).

### **Face Swap Endpoints**

#### **GET /api/templates**
List available character templates.
```json
{
  "success": true,
  "templates": [
    {
      "name": "superhero.jpg",
      "path": "/static/templates/superhero.jpg"
    }
  ],
  "count": 4
}
```

#### **POST /api/swap**
Generate face swap (requires authentication).
- **Form Data**:
  - `template_name`: Template filename
  - `webcam`: Image file from camera
  - `apply_frame`: Boolean for frame overlay
- **Response**:
```json
{
  "success": true,
  "message": "Face swap berhasil dilakukan",
  "data": {
    "result_url": "/static/results/result_xxx.png",
    "result_filename": "result_xxx.png",
    "template_used": "superhero.jpg",
    "faces_detected": {"source": 1, "template": 1},
    "frame_applied": true,
    "processing_time": "2024-01-01T12:00:00"
  }
}
```

### **Utility Endpoints**

#### **GET /api/history**
Get user's generation history (requires authentication).

#### **DELETE /api/results/{filename}**
Delete specific result file (requires authentication).

#### **GET /health**
Application health check.

#### **GET /api/info**
Application information and available endpoints.

---

## 🏗️ **Architecture**

### **Project Structure**
```
ai-face-swap-studio/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # Documentation
├── face_swap.db           # SQLite database
│
├── pages/                 # Frontend HTML pages
│   ├── login.html         # Authentication page
│   ├── character.html     # Template selection
│   ├── camera.html        # Photo capture
│   └── result.html        # Results & download
│
├── static/                # Static files
│   ├── uploads/           # Temporary uploads
│   ├── templates/         # Character templates
│   ├── results/           # Generated results
│   └── images/            # Frame overlays
│
└── models/                # AI models (auto-downloaded)
    ├── buffalo_l/         # Face detection model
    └── inswapper_128.onnx # Face swapping model
```

### **Technology Stack**

#### **Backend**
- **FastAPI**: Modern, fast web framework
- **SQLite**: Lightweight database
- **InsightFace**: Face detection and swapping
- **OpenCV**: Image processing
- **JWT**: Token-based authentication

#### **Frontend**
- **Vanilla JavaScript**: No framework dependencies
- **HTML5**: Modern web standards
- **CSS3**: Responsive design with animations
- **QRCode.js**: QR code generation
- **Font Awesome**: Icon library

#### **AI/ML**
- **InsightFace**: Face analysis and recognition
- **ONNX Runtime**: Model inference
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations

---

## 🎨 **Frontend**

### **Design System**
- **Color Scheme**: Dark blue to pink gradient background
- **Accent Color**: Cyan (#00ffcc) for highlights
- **Typography**: Poppins font family
- **Effects**: Glassmorphism, smooth animations

### **Components**
- **Responsive Layouts**: Mobile-first design
- **Interactive Buttons**: Hover and press effects
- **Loading States**: Spinners and progress indicators
- **Error Handling**: User-friendly error messages

### **Pages**

#### **Login Page** (`/login`)
- Authentication form
- Registration option
- Auto-redirect if logged in

#### **Character Page** (`/character`)
- Grid layout for templates
- Selection with visual feedback
- Template preview

#### **Camera Page** (`/camera`)
- Real-time camera stream
- Auto countdown timer
- Face positioning guide
- Auto processing

#### **Result Page** (`/result`)
- Generated image display
- QR code for download
- Multiple download options
- Social sharing

---

## 🔒 **Security**

### **Authentication Security**
- **Password Hashing**: PBKDF2 with salt (100,000 iterations)
- **JWT Tokens**: Secure session management
- **Token Expiration**: 24-hour default lifetime
- **Input Validation**: Server-side validation

### **File Security**
- **Upload Validation**: File type and size checks
- **MIME Type Verification**: Prevent malicious uploads
- **Temporary Storage**: Auto-cleanup of uploaded files
- **Path Sanitization**: Prevent directory traversal

### **API Security**
- **Authentication Required**: Protected endpoints
- **CORS Configuration**: Controlled cross-origin requests
- **Rate Limiting**: (Recommended for production)
- **Input Sanitization**: SQL injection prevention

### **Data Privacy**
- **Temporary Files**: Auto-deletion after processing
- **User Isolation**: User-specific data access
- **No Data Retention**: Photos not permanently stored
- **GDPR Compliance**: User data control

---

## 📊 **Performance**

### **Optimization Features**
- **Async Operations**: Non-blocking file I/O
- **Model Caching**: Models loaded once at startup
- **Resource Cleanup**: Automatic temporary file removal
- **Efficient Processing**: Optimized image operations

### **Performance Metrics**
- **Face Detection**: ~2-3 seconds
- **Face Swapping**: ~5-10 seconds
- **Total Process Time**: ~15-30 seconds
- **Memory Usage**: ~2-4GB (with models loaded)

### **Scalability**
- **Multiple Workers**: Uvicorn/Gunicorn support
- **Load Balancing**: Stateless design
- **Database Optimization**: Indexed queries
- **CDN Support**: Static file serving

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Installation Problems**
```bash
# OpenCV installation error
pip install opencv-python-headless

# InsightFace build error
pip install cmake
pip install insightface

# CUDA compatibility (optional)
pip install onnxruntime-gpu
```

#### **Runtime Errors**
- **Camera Access Denied**: Check browser permissions
- **Model Download Failed**: Check internet connection
- **QR Code Not Generated**: Multiple fallbacks implemented
- **Face Not Detected**: Ensure good lighting and positioning

#### **Performance Issues**
- **Slow Processing**: Reduce `DET_SIZE` in configuration
- **High Memory Usage**: Use CPU mode (`CTX_ID = -1`)
- **Long Startup Time**: Models loading on first run

### **Debug Mode**
```bash
# Enable debug logging
export DEBUG=True
python main.py

# Check logs
tail -f logs/app.log
```

### **Health Checks**
- **Application**: GET `/health`
- **Database**: Check `face_swap.db` exists
- **Models**: Verify `inswapper_128.onnx` downloaded
- **Frontend**: Test all page routes

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Fork repository
git clone <your-fork>
cd ai-face-swap-studio

# Create feature branch
git checkout -b feature/your-feature

# Install development dependencies
pip install -r requirements-dev.txt

# Make changes and test
python main.py

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature
```

### **Contribution Guidelines**
- Follow Python PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure backward compatibility
- Test on multiple platforms

### **Reporting Issues**
- Use GitHub Issues
- Include error logs
- Provide reproduction steps
- Specify environment details

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **Third-Party Licenses**
- **InsightFace**: Apache 2.0 License
- **FastAPI**: MIT License
- **OpenCV**: BSD License
- **Font Awesome**: Font Awesome License

---

## 🙏 **Acknowledgments**

- **InsightFace Team**: For excellent face analysis framework
- **FastAPI**: For modern Python web framework
- **OpenCV Community**: For computer vision tools
- **Contributors**: Thanks to all contributors

---

## 📞 **Support**

- **Documentation**: This README
- **API Docs**: http://localhost:5000/docs
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

### **Version History**
- **v2.0.0**: Auto countdown camera, QR codes, clean routing
- **v1.0.0**: Basic face swap with authentication
- **v0.1.0**: Initial prototype

---

**Made with ❤️ by the AI Face Swap Studio Team**

*Transform yourself with AI magic! 🎭✨*