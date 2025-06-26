from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import logging
import aiofiles
import asyncio
from contextlib import asynccontextmanager
import uuid
import mimetypes

# Import authentication modules
import sqlite3
import hashlib
import secrets
from datetime import timedelta
import jwt
from pydantic import BaseModel

# Pydantic models for authentication
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# Configuration
class Config:
    UPLOAD_DIR = Path("static/uploads")
    TEMPLATE_DIR = Path("static/templates")
    RESULT_DIR = Path("static/results")
    FRAME_DIR = Path("static/images")
    PAGES_DIR = Path("pages")  # Frontend pages directory
    
    # AR Photo directories (added for compatibility)
    AR_ASSETS_DIR = Path("static/ar_assets")
    COUNTDOWN_DIR = Path("static/ar_assets/countdown")
    AR_RESULTS_DIR = Path("static/ar_results")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/bmp'}
    
    # Face detection parameters
    DET_SIZE = (640, 640)
    CTX_ID = 0
    
    # JWT Configuration
    JWT_SECRET_KEY = secrets.token_urlsafe(32)
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables untuk model
face_app = None
swapper = None

# Database setup
DB_PATH = Path("face_swap.db")

class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_swap_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    template_name TEXT NOT NULL,
                    result_filename TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_history ON face_swap_history(user_id)")
            
            conn.commit()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

class AuthService:
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash
    
    def create_jwt_token(self, user_id: int, username: str) -> str:
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)
        return token
    
    def verify_jwt_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def register_user(self, user_data: UserCreate) -> dict:
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (user_data.username,)
            )
            
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username sudah digunakan"
                )
            
            if len(user_data.password) < 4:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password minimal 4 karakter"
                )
            
            password_hash, salt = self.hash_password(user_data.password)
            
            cursor = conn.execute("""
                INSERT INTO users (username, password_hash, salt)
                VALUES (?, ?, ?)
            """, (user_data.username, password_hash, salt))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {
                "success": True,
                "message": "User berhasil didaftarkan",
                "user_id": user_id
            }
    
    def login_user(self, login_data: UserLogin) -> dict:
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, username, password_hash, salt, is_active
                FROM users WHERE username = ?
            """, (login_data.username,))
            
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Username atau password salah"
                )
            
            user_id, username, password_hash, salt, is_active = user
            
            if not is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Akun tidak aktif"
                )
            
            if not self.verify_password(login_data.password, password_hash, salt):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Username atau password salah"
                )
            
            token = self.create_jwt_token(user_id, username)
            
            conn.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user_id,))
            
            conn.commit()
            
            return {
                "success": True,
                "message": "Login berhasil",
                "token": token,
                "user": {
                    "id": user_id,
                    "username": username
                }
            }
    
    def get_user_by_token(self, token: str) -> dict:
        payload = self.verify_jwt_token(token)
        user_id = payload.get("user_id")
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, username, created_at, last_login
                FROM users WHERE id = ? AND is_active = TRUE
            """, (user_id,))
            
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User tidak ditemukan"
                )
            
            return {
                "id": user[0],
                "username": user[1],
                "created_at": user[2],
                "last_login": user[3]
            }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global face_app, swapper
    
    try:
        logger.info("Initializing face analysis model...")
        face_app = FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=Config.CTX_ID, det_size=Config.DET_SIZE)
        
        logger.info("Loading face swapper model...")
        swapper = insightface.model_zoo.get_model(
            'inswapper_128.onnx', 
            download=False, 
            download_zip=False
        )
        
        # Create all directories
        directories = [
            Config.UPLOAD_DIR, Config.TEMPLATE_DIR, Config.RESULT_DIR, 
            Config.FRAME_DIR, Config.PAGES_DIR, Config.AR_ASSETS_DIR, 
            Config.COUNTDOWN_DIR, Config.AR_RESULTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize AR Photo module
        try:
            from ar_photo import init_ar_photo
            init_ar_photo()
            logger.info("AR Photo module initialized successfully")
        except ImportError:
            logger.warning("AR Photo module not found, skipping AR initialization")
        except Exception as e:
            logger.error(f"Failed to initialize AR Photo module: {e}")
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        logger.info("Application shutdown")

# Initialize FastAPI app
app = FastAPI(
    title="AI Face Swap Studio",
    description="Advanced Face Swapping API with Authentication and AR Photo",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
auth_service = AuthService()
security = HTTPBearer(auto_error=False)

# Custom exceptions
class FaceSwapError(Exception):
    pass

class ValidationError(Exception):
    pass

# Authentication dependency
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    token = credentials.credentials
    return auth_service.get_user_by_token(token)

# Utility functions
def validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise ValidationError("Filename tidak boleh kosong")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Ekstensi file tidak didukung. Gunakan: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )
    
    mime_type = mimetypes.guess_type(file.filename)[0]
    if mime_type not in Config.ALLOWED_MIME_TYPES:
        raise ValidationError(f"Tipe file tidak didukung: {mime_type}")

def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_ext = Path(original_filename).suffix.lower()
    
    safe_filename = f"{prefix}{timestamp}_{unique_id}{file_ext}"
    return safe_filename

async def save_uploaded_file(file: UploadFile, save_path: Path) -> Path:
    try:
        async with aiofiles.open(save_path, 'wb') as f:
            content = await file.read()
            
            if len(content) > Config.MAX_FILE_SIZE:
                raise ValidationError(
                    f"File terlalu besar. Maksimal {Config.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            await f.write(content)
        
        logger.info(f"File saved: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error saving file {save_path}: {e}")
        if save_path.exists():
            save_path.unlink()
        raise

def detect_faces(image_path: Path) -> List:
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar: {image_path}")
        
        faces = face_app.get(img)
        logger.info(f"Detected {len(faces)} faces in {image_path}")
        return faces
    
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        raise FaceSwapError(f"Error dalam deteksi wajah: {e}")

def swap_faces(src_path: Path, dst_path: Path, output_path: Path) -> Path:
    try:
        img_src = cv2.imread(str(src_path))
        img_dst = cv2.imread(str(dst_path))
        
        if img_src is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar sumber: {src_path}")
        if img_dst is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar template: {dst_path}")
        
        faces_src = detect_faces(src_path)
        faces_dst = detect_faces(dst_path)
        
        if len(faces_src) == 0:
            raise FaceSwapError("Tidak ada wajah yang terdeteksi pada gambar sumber")
        if len(faces_dst) == 0:
            raise FaceSwapError("Tidak ada wajah yang terdeteksi pada template")
        
        face_src = faces_src[0]
        face_dst = faces_dst[0]
        
        result = swapper.get(img_dst.copy(), face_dst, face_src, paste_back=True)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), result)
        
        if not success:
            raise FaceSwapError("Gagal menyimpan hasil face swap")
        
        logger.info(f"Face swap completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Face swap error: {e}")
        if isinstance(e, FaceSwapError):
            raise
        raise FaceSwapError(f"Error dalam proses face swap: {e}")

def overlay_frame(base_image_path: Path, frame_path: Path, output_path: Path) -> Path:
    try:
        if not frame_path.exists():
            logger.warning(f"Frame file not found: {frame_path}")
            return base_image_path
        
        base_img = cv2.imread(str(base_image_path), cv2.IMREAD_UNCHANGED)
        frame_img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        
        if base_img is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar dasar: {base_image_path}")
        if frame_img is None:
            logger.warning(f"Cannot read frame image: {frame_path}")
            return base_image_path
        
        frame_img = cv2.resize(frame_img, (base_img.shape[1], base_img.shape[0]))
        
        if frame_img.shape[2] == 4:
            alpha_mask = frame_img[:, :, 3] / 255.0
            alpha_mask = np.stack([alpha_mask] * 3, axis=2)
            
            base_img_rgb = base_img[:, :, :3]
            frame_img_rgb = frame_img[:, :, :3]
            
            result = (1 - alpha_mask) * base_img_rgb + alpha_mask * frame_img_rgb
            result = result.astype(np.uint8)
        else:
            result = cv2.addWeighted(base_img, 0.7, frame_img, 0.3, 0)
        
        success = cv2.imwrite(str(output_path), result)
        if not success:
            raise FaceSwapError("Gagal menyimpan hasil overlay frame")
        
        logger.info(f"Frame overlay completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Frame overlay error: {e}")
        return base_image_path

def serve_html_page(page_name: str) -> HTMLResponse:
    """Serve HTML page from pages directory"""
    try:
        page_path = Config.PAGES_DIR / f"{page_name}.html"
        if page_path.exists():
            with open(page_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            # Return a basic template if file doesn't exist
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html><head><title>Page Not Found</title></head>
            <body>
            <h1>Page: {page_name}</h1>
            <p>File {page_name}.html tidak ditemukan di folder pages/</p>
            <p>Silakan letakkan file HTML di folder pages/</p>
            <p><a href="/login">Go to Login</a></p>
            </body></html>
            """)
    except Exception as e:
        logger.error(f"Error serving page {page_name}: {e}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html><head><title>Error</title></head>
        <body>
        <h1>Error</h1>
        <p>Terjadi kesalahan saat memuat halaman {page_name}</p>
        <p>Error: {str(e)}</p>
        </body></html>
        """)

# ===== FRONTEND ROUTES =====

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to login page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><head><title>AI Face Swap Studio</title>
    <meta http-equiv="refresh" content="0; url=/login">
    </head>
    <body>
    <p>Redirecting to <a href="/login">Login Page</a>...</p>
    </body></html>
    """)

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Serve login page"""
    return serve_html_page("login")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve main dashboard/menu page"""
    return serve_html_page("dashboard")

@app.get("/character", response_class=HTMLResponse)
async def character_page():
    """Serve character selection page"""
    return serve_html_page("character")

@app.get("/camera", response_class=HTMLResponse)
async def camera_page():
    """Serve camera page"""
    return serve_html_page("camera")

@app.get("/ar-character", response_class=HTMLResponse)
async def ar_character_page():
    """Serve AR character selection page"""
    return serve_html_page("ar_character")

@app.get("/ar-camera", response_class=HTMLResponse)
async def ar_camera_page():
    """Serve AR camera page"""
    return serve_html_page("ar_camera")

@app.get("/result", response_class=HTMLResponse)
async def result_page():
    """Serve result page"""
    return serve_html_page("result")

@app.get("/index", response_class=HTMLResponse)
async def index_page():
    """Serve index page or redirect to dashboard"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><head><title>AI Face Swap Studio</title>
    <meta http-equiv="refresh" content="0; url=/dashboard">
    </head>
    <body>
    <p>Redirecting to <a href="/dashboard">Dashboard</a>...</p>
    </body></html>
    """)

# ===== API ROUTES =====

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Face Swap Studio is running"}

@app.get("/api/info")
async def app_info():
    """Get application information"""
    return {
        "app_name": "AI Face Swap Studio",
        "version": "2.1.0",
        "description": "Advanced Face Swapping API with Authentication and AR Photo",
        "pages": [
            {"path": "/login", "description": "Login page"},
            {"path": "/dashboard", "description": "Main menu dashboard"},
            {"path": "/character", "description": "Character selection"},
            {"path": "/camera", "description": "Photo capture"},
            {"path": "/ar-character", "description": "AR Character selection"},
            {"path": "/ar-camera", "description": "AR Photo capture"},
            {"path": "/result", "description": "Result with QR code"}
        ],
        "features": [
            "Face Swapping",
            "AR Photo Overlay",
            "User Authentication",
            "QR Code Generation",
            "User Dashboard"
        ],
        "api_docs": "/docs"
    }

# Authentication endpoints
@app.post("/api/register")
async def register(user_data: UserCreate):
    return auth_service.register_user(user_data)

@app.post("/api/login")
async def login(login_data: UserLogin):
    return auth_service.login_user(login_data)

@app.post("/api/logout")
async def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Logout user and invalidate session"""
    if not credentials:
        return {"success": True, "message": "Logout berhasil"}
    
    try:
        token = credentials.credentials
        # You can add token invalidation logic here if needed
        # For now, we rely on client-side token removal
        return {
            "success": True,
            "message": "Logout berhasil"
        }
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {"success": True, "message": "Logout berhasil"}

@app.get("/api/me")
async def get_me(current_user = Depends(get_current_user)):
    return {
        "success": True,
        "user": current_user
    }

# Template management
@app.get("/api/templates")
async def list_templates():
    try:
        templates = []
        if Config.TEMPLATE_DIR.exists():
            for file_path in Config.TEMPLATE_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in Config.ALLOWED_EXTENSIONS:
                    templates.append({
                        "name": file_path.name,
                        "path": f"/static/templates/{file_path.name}"
                    })
        
        # If no templates exist, create sample data
        if not templates:
            sample_templates = [
                {"name": "superhero.jpg", "path": "/static/templates/superhero.jpg"},
                {"name": "princess.jpg", "path": "/static/templates/princess.jpg"},
                {"name": "warrior.jpg", "path": "/static/templates/warrior.jpg"},
                {"name": "cartoon.jpg", "path": "/static/templates/cartoon.jpg"}
            ]
            templates = sample_templates
        
        return JSONResponse({
            "success": True,
            "templates": templates,
            "count": len(templates)
        })
    
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mengambil daftar template"
        )

# Face swap endpoint
@app.post("/api/swap")
async def swap_faces_api(
    template_name: str = Form(...),
    webcam: UploadFile = File(...),
    source: Optional[UploadFile] = File(None),
    apply_frame: bool = Form(True),
    current_user = Depends(get_current_user)
):
    temp_files = []
    
    try:
        validate_file(webcam)
        if source:
            validate_file(source)
        
        template_path = Config.TEMPLATE_DIR / template_name
        if not template_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template '{template_name}' tidak ditemukan"
            )
        
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        webcam_filename = generate_unique_filename(webcam.filename, "webcam_")
        webcam_path = Config.UPLOAD_DIR / webcam_filename
        await save_uploaded_file(webcam, webcam_path)
        temp_files.append(webcam_path)
        
        source_path = webcam_path
        if source:
            source_filename = generate_unique_filename(source.filename, "source_")
            source_path = Config.UPLOAD_DIR / source_filename
            await save_uploaded_file(source, source_path)
            temp_files.append(source_path)
        
        result_filename = f"result_{timestamp}_{unique_id}.png"
        result_path = Config.RESULT_DIR / result_filename
        
        logger.info(f"Starting face swap: {source_path} -> {template_path}")
        swap_result_path = swap_faces(source_path, template_path, result_path)
        
        final_result_path = swap_result_path
        if apply_frame:
            frame_path = Config.FRAME_DIR / "frame1.png"
            final_result_path = overlay_frame(swap_result_path, frame_path, result_path)
        
        # Save to history
        with auth_service.db_manager.get_connection() as conn:
            conn.execute("""
                INSERT INTO face_swap_history (user_id, template_name, result_filename)
                VALUES (?, ?, ?)
            """, (current_user["id"], template_name, result_filename))
            conn.commit()
        
        response_data = {
            "success": True,
            "message": "Face swap berhasil dilakukan",
            "data": {
                "result_url": f"/static/results/{result_filename}",
                "result_filename": result_filename,
                "template_used": template_name,
                "faces_detected": {
                    "source": len(detect_faces(source_path)),
                    "template": len(detect_faces(template_path))
                },
                "frame_applied": apply_frame,
                "processing_time": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Face swap successful: {final_result_path}")
        return JSONResponse(response_data)
    
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except FaceSwapError as e:
        logger.error(f"Face swap error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in face swap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan internal pada server"
        )
    
    finally:
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")

# User history
@app.get("/api/history")
async def get_user_history(current_user = Depends(get_current_user)):
    try:
        with auth_service.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT template_name, result_filename, created_at
                FROM face_swap_history 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 50
            """, (current_user["id"],))
            
            history = cursor.fetchall()
            
            history_list = []
            for record in history:
                history_list.append({
                    "template_name": record[0],
                    "result_filename": record[1],
                    "result_url": f"/static/results/{record[1]}",
                    "created_at": record[2]
                })
            
            return {
                "success": True,
                "history": history_list,
                "count": len(history_list)
            }
    
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mengambil riwayat"
        )

@app.delete("/api/results/{filename}")
async def delete_result(filename: str, current_user = Depends(get_current_user)):
    try:
        result_path = Config.RESULT_DIR / filename
        
        if not result_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File hasil tidak ditemukan"
            )
        
        # Verify ownership
        with auth_service.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id FROM face_swap_history 
                WHERE user_id = ? AND result_filename = ?
            """, (current_user["id"], filename))
            
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Tidak memiliki akses ke file ini"
                )
        
        result_path.unlink()
        logger.info(f"Deleted result file: {result_path}")
        
        return JSONResponse({
            "success": True,
            "message": f"File '{filename}' berhasil dihapus"
        })
    
    except Exception as e:
        logger.error(f"Error deleting result file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal menghapus file"
        )

# Add this to main.py - Replace the fallback AR endpoints

@app.get("/api/ar/characters")
async def ar_characters_dynamic():
    """Dynamic AR characters from directory scan - FIXED VERSION"""
    try:
        from pathlib import Path
        
        characters = []
        
        # Define directories  
        thumbnail_dir = Path("static/ar_assets/thumbnail")
        ar_assets_dir = Path("static/ar_assets")
        
        # Create directories if they don't exist
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        ar_assets_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan thumbnail directory for real images
        if thumbnail_dir.exists():
            for thumbnail_file in thumbnail_dir.iterdir():
                if thumbnail_file.is_file() and thumbnail_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    character_name = thumbnail_file.stem
                    
                    # Look for corresponding webm/mp4 animation file
                    animation_file = None
                    animation_url = None
                    has_animation = False
                    
                    # Check for animation files (prioritize webm for AR)
                    for ext in ['.webm', '.mp4', '.mov']:
                        potential_animation = ar_assets_dir / f"{character_name}{ext}"
                        if potential_animation.exists():
                            animation_file = potential_animation
                            animation_url = f"/static/ar_assets/{character_name}{ext}"
                            has_animation = True
                            logger.info(f"Found animation for {character_name}: {ext}")
                            break
                    
                    # IMPORTANT: Always prefer webm for AR experience
                    webm_file = ar_assets_dir / f"{character_name}.webm"
                    if webm_file.exists():
                        animation_file = webm_file
                        animation_url = f"/static/ar_assets/{character_name}.webm"
                        has_animation = True
                    
                    character_data = {
                        "name": character_name,
                        "display_name": character_name.replace('_', ' ').title(),
                        "thumbnail": f"/static/ar_assets/thumbnail/{thumbnail_file.name}",
                        "has_animation": has_animation,
                        "animation_url": animation_url,
                        "webm_url": f"/static/ar_assets/{character_name}.webm" if webm_file.exists() else None,
                        "thumbnail_path": str(thumbnail_file),
                        "animation_path": str(animation_file) if animation_file else None,
                        "type": "photo_ar"  # Not face swap!
                    }
                    
                    characters.append(character_data)
                    logger.info(f"AR Character found: {character_name} (webm: {webm_file.exists()})")
        
        # Check if we have required AR assets
        webm_characters = [c for c in characters if c['webm_url']]
        
        if not characters:
            return {
                "success": True,
                "characters": [],
                "count": 0,
                "mode": "EMPTY_DIRECTORY",
                "message": "No AR characters found. Add thumbnail + webm files.",
                "instructions": {
                    "steps": [
                        "1. Add character thumbnail: static/ar_assets/thumbnail/superhero.png",
                        "2. Add character animation: static/ar_assets/superhero.webm",
                        "3. Refresh page to see characters"
                    ],
                    "required_files": [
                        "Thumbnail: .png/.jpg/.jpeg (for display)",
                        "Animation: .webm (for AR capture)"
                    ]
                }
            }
        
        return {
            "success": True,
            "characters": characters,
            "count": len(characters),
            "webm_count": len(webm_characters),
            "mode": "DIRECTORY_SCAN", 
            "message": f"Found {len(characters)} AR characters ({len(webm_characters)} with WebM animations)"
        }
    
    except Exception as e:
        logger.error(f"Error scanning AR characters: {e}")
        return {
            "success": False,
            "error": str(e),
            "characters": [],
            "count": 0
        }

@app.post("/api/ar/upload-character")
async def upload_character(
    name: str = Form(...),
    thumbnail: UploadFile = File(...),
    animation: Optional[UploadFile] = File(None),
    current_user = Depends(get_current_user)
):
    """Upload character thumbnail and animation"""
    try:
        from pathlib import Path
        import aiofiles
        
        # Validate inputs
        if not name or not name.strip():
            raise HTTPException(status_code=400, detail="Character name is required")
        
        # Sanitize character name
        import re
        safe_name = re.sub(r'[^\w\-_]', '', name.strip().lower())
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid character name")
        
        # Create directories
        thumbnail_dir = Path("static/ar_assets/thumbnail")
        ar_assets_dir = Path("static/ar_assets")
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        ar_assets_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate thumbnail
        if not thumbnail.filename:
            raise HTTPException(status_code=400, detail="Thumbnail file is required")
        
        thumbnail_ext = Path(thumbnail.filename).suffix.lower()
        if thumbnail_ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            raise HTTPException(status_code=400, detail="Invalid thumbnail format")
        
        # Save thumbnail
        thumbnail_filename = f"{safe_name}{thumbnail_ext}"
        thumbnail_path = thumbnail_dir / thumbnail_filename
        
        async with aiofiles.open(thumbnail_path, 'wb') as f:
            content = await thumbnail.read()
            await f.write(content)
        
        # Save animation if provided
        animation_path = None
        if animation and animation.filename:
            animation_ext = Path(animation.filename).suffix.lower()
            if animation_ext not in ['.webm', '.mp4', '.mov']:
                raise HTTPException(status_code=400, detail="Invalid animation format")
            
            animation_filename = f"{safe_name}{animation_ext}"
            animation_path = ar_assets_dir / animation_filename
            
            async with aiofiles.open(animation_path, 'wb') as f:
                content = await animation.read()
                await f.write(content)
        
        logger.info(f"Character uploaded: {safe_name}")
        
        return {
            "success": True,
            "message": f"Character '{safe_name}' uploaded successfully",
            "character": {
                "name": safe_name,
                "thumbnail": f"/static/ar_assets/thumbnail/{thumbnail_filename}",
                "has_animation": animation_path is not None,
                "animation_url": f"/static/ar_assets/{animation_path.name}" if animation_path else None
            }
        }
    
    except Exception as e:
        logger.error(f"Error uploading character: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.delete("/api/ar/characters/{character_name}")
async def delete_character(character_name: str, current_user = Depends(get_current_user)):
    """Delete character files"""
    try:
        from pathlib import Path
        import re
        
        # Sanitize name
        safe_name = re.sub(r'[^\w\-_]', '', character_name.strip().lower())
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid character name")
        
        thumbnail_dir = Path("static/ar_assets/thumbnail")
        ar_assets_dir = Path("static/ar_assets")
        
        deleted_files = []
        
        # Delete thumbnail
        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            thumbnail_path = thumbnail_dir / f"{safe_name}{ext}"
            if thumbnail_path.exists():
                thumbnail_path.unlink()
                deleted_files.append(str(thumbnail_path))
        
        # Delete animation
        for ext in ['.webm', '.mp4', '.mov']:
            animation_path = ar_assets_dir / f"{safe_name}{ext}"
            if animation_path.exists():
                animation_path.unlink()
                deleted_files.append(str(animation_path))
        
        if not deleted_files:
            raise HTTPException(status_code=404, detail="Character not found")
        
        logger.info(f"Character deleted: {safe_name}")
        
        return {
            "success": True,
            "message": f"Character '{safe_name}' deleted successfully",
            "deleted_files": deleted_files
        }
    
    except Exception as e:
        logger.error(f"Error deleting character: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/api/ar/test")  
async def ar_test_dynamic():
    """Test AR endpoints"""
    return {
        "success": True,
        "message": "AR endpoints working (dynamic directory scan)",
        "mode": "DYNAMIC",
        "endpoints": [
            "/api/ar/characters - List characters from directory",
            "/api/ar/scan - Debug directory contents", 
            "/api/ar/upload-character - Upload new character",
            "/api/ar/characters/{name} - Delete character",
            "/api/ar/test - This endpoint"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="localhost",
        port=5000,
        reload=True,
        log_level="info"
    )