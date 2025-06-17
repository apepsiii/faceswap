from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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

# Configuration
class Config:
    UPLOAD_DIR = Path("static/uploads")
    TEMPLATE_DIR = Path("static/templates")
    RESULT_DIR = Path("static/results")
    FRAME_DIR = Path("static/images")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/bmp'}
    
    # Face detection parameters
    DET_SIZE = (640, 640)
    CTX_ID = 0

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables untuk model
face_app = None
swapper = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global face_app, swapper
    
    try:
        # Initialize models
        logger.info("Initializing face analysis model...")
        face_app = FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=Config.CTX_ID, det_size=Config.DET_SIZE)
        
        logger.info("Loading face swapper model...")
        swapper = insightface.model_zoo.get_model(
            'inswapper_128.onnx', 
            download=False, 
            download_zip=False
        )
        
        # Create directories
        for directory in [Config.UPLOAD_DIR, Config.TEMPLATE_DIR, 
                         Config.RESULT_DIR, Config.FRAME_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        logger.info("Application shutdown")

# Initialize FastAPI app
app = FastAPI(
    title="Face Swap API",
    description="API untuk melakukan face swapping dengan template",
    version="2.0.0",
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

class FaceSwapError(Exception):
    """Custom exception for face swap operations"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise ValidationError("Filename tidak boleh kosong")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Ekstensi file tidak didukung. Gunakan: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )
    
    # Check MIME type
    mime_type = mimetypes.guess_type(file.filename)[0]
    if mime_type not in Config.ALLOWED_MIME_TYPES:
        raise ValidationError(f"Tipe file tidak didukung: {mime_type}")

def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """Generate unique filename with timestamp and UUID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_ext = Path(original_filename).suffix.lower()
    
    safe_filename = f"{prefix}{timestamp}_{unique_id}{file_ext}"
    return safe_filename

async def save_uploaded_file(file: UploadFile, save_path: Path) -> Path:
    """Save uploaded file asynchronously"""
    try:
        async with aiofiles.open(save_path, 'wb') as f:
            content = await file.read()
            
            # Validate file size
            if len(content) > Config.MAX_FILE_SIZE:
                raise ValidationError(
                    f"File terlalu besar. Maksimal {Config.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            await f.write(content)
        
        logger.info(f"File saved: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error saving file {save_path}: {e}")
        # Cleanup on error
        if save_path.exists():
            save_path.unlink()
        raise

def detect_faces(image_path: Path) -> List:
    """Detect faces in image"""
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
    """Perform face swapping"""
    try:
        # Read images
        img_src = cv2.imread(str(src_path))
        img_dst = cv2.imread(str(dst_path))
        
        if img_src is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar sumber: {src_path}")
        if img_dst is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar template: {dst_path}")
        
        # Detect faces
        faces_src = detect_faces(src_path)
        faces_dst = detect_faces(dst_path)
        
        if len(faces_src) == 0:
            raise FaceSwapError("Tidak ada wajah yang terdeteksi pada gambar sumber")
        if len(faces_dst) == 0:
            raise FaceSwapError("Tidak ada wajah yang terdeteksi pada template")
        
        # Use the first detected face from each image
        face_src = faces_src[0]
        face_dst = faces_dst[0]
        
        # Perform face swap
        result = swapper.get(img_dst.copy(), face_dst, face_src, paste_back=True)
        
        # Save result
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
    """Overlay frame on image"""
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
        
        # Resize frame to match base image
        frame_img = cv2.resize(frame_img, (base_img.shape[1], base_img.shape[0]))
        
        # Apply overlay
        if frame_img.shape[2] == 4:  # PNG with alpha channel
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
        return base_image_path  # Return original if overlay fails

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Face Swap API is running"}

@app.get("/api/templates")
async def list_templates():
    """List available templates"""
    try:
        templates = []
        if Config.TEMPLATE_DIR.exists():
            for file_path in Config.TEMPLATE_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in Config.ALLOWED_EXTENSIONS:
                    templates.append({
                        "name": file_path.name,
                        "path": f"/static/templates/{file_path.name}"
                    })
        
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

@app.post("/api/swap")
async def swap_faces_api(
    template_name: str = Form(..., description="Nama file template"),
    webcam: UploadFile = File(..., description="File gambar dari webcam"),
    source: Optional[UploadFile] = File(None, description="File gambar sumber tambahan (opsional)"),
    apply_frame: bool = Form(True, description="Aplikasikan frame overlay")
):
    """
    Perform face swapping between source image and template
    
    - **template_name**: Nama file template yang tersedia
    - **webcam**: File gambar dari webcam (wajib)
    - **source**: File gambar sumber tambahan (opsional, akan menimpa webcam jika ada)
    - **apply_frame**: Aplikasikan frame overlay pada hasil
    """
    temp_files = []  # Track temporary files for cleanup
    
    try:
        # Validate inputs
        validate_file(webcam)
        if source:
            validate_file(source)
        
        # Check template exists
        template_path = Config.TEMPLATE_DIR / template_name
        if not template_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template '{template_name}' tidak ditemukan"
            )
        
        # Generate unique filenames
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save webcam file
        webcam_filename = generate_unique_filename(webcam.filename, "webcam_")
        webcam_path = Config.UPLOAD_DIR / webcam_filename
        await save_uploaded_file(webcam, webcam_path)
        temp_files.append(webcam_path)
        
        # Determine source file (use source if provided, otherwise use webcam)
        source_path = webcam_path
        if source:
            source_filename = generate_unique_filename(source.filename, "source_")
            source_path = Config.UPLOAD_DIR / source_filename
            await save_uploaded_file(source, source_path)
            temp_files.append(source_path)
        
        # Generate result filename
        result_filename = f"result_{timestamp}_{unique_id}.png"
        result_path = Config.RESULT_DIR / result_filename
        
        # Perform face swap
        logger.info(f"Starting face swap: {source_path} -> {template_path}")
        swap_result_path = swap_faces(source_path, template_path, result_path)
        
        # Apply frame overlay if requested
        final_result_path = swap_result_path
        if apply_frame:
            frame_path = Config.FRAME_DIR / "frame1.png"
            final_result_path = overlay_frame(swap_result_path, frame_path, result_path)
        
        # Build response
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
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")

@app.delete("/api/results/{filename}")
async def delete_result(filename: str):
    """Delete a result file"""
    try:
        result_path = Config.RESULT_DIR / filename
        
        if not result_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File hasil tidak ditemukan"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )