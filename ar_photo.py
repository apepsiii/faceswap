# ar_photo.py
"""
AR Photo functionality module
Separated from main.py for better code organization
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import logging
import uuid
import qrcode
from io import BytesIO
import base64
import mimetypes
import aiofiles
import shutil

# Setup logging
logger = logging.getLogger(__name__)

# AR Photo specific configuration
class ARConfig:
    AR_ASSETS_DIR = Path("static/ar_assets")
    COUNTDOWN_DIR = Path("static/ar_assets/countdown")
    THUMBNAIL_DIR = Path("static/ar_assets/thumbnail")
    AR_RESULTS_DIR = Path("static/ar_results")
    UPLOAD_DIR = Path("static/uploads")
    FRAME_DIR = Path("static/images")
    
    # AR specific settings
    WEBM_DURATION = 3  # seconds for webm character display
    AR_COUNTDOWN_DURATION = 5  # seconds for countdown
    RESULT_WAIT_DURATION = 5  # seconds before redirect to result
    QR_CODE_SIZE = 10
    QR_BORDER = 4
    
    # File settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/bmp'}
    
    # Domain for QR codes (update this for production)
    DOMAIN_URL = "http://localhost:5000"

# Create router for AR Photo endpoints
router = APIRouter(prefix="/api/ar", tags=["AR Photo"])

# Authentication setup (avoiding circular import)
security = HTTPBearer(auto_error=False)

class ARPhotoError(Exception):
    """Custom exception for AR Photo operations"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

# Utility functions (copied from main to avoid circular import)
def validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise ValidationError("Filename tidak boleh kosong")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ARConfig.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Ekstensi file tidak didukung. Gunakan: {', '.join(ARConfig.ALLOWED_EXTENSIONS)}"
        )
    
    mime_type = mimetypes.guess_type(file.filename)[0]
    if mime_type not in ARConfig.ALLOWED_MIME_TYPES:
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
            
            if len(content) > ARConfig.MAX_FILE_SIZE:
                raise ValidationError(
                    f"File terlalu besar. Maksimal {ARConfig.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            await f.write(content)
        
        logger.info(f"File saved: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error saving file {save_path}: {e}")
        if save_path.exists():
            save_path.unlink()
        raise

def generate_qr_code(data: str, size: int = ARConfig.QR_CODE_SIZE) -> str:
    """Generate QR code and return as base64 string"""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=size,
            border=ARConfig.QR_BORDER,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = BytesIO()
        qr_img.save(buffered, format="PNG")
        qr_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{qr_base64}"
    
    except Exception as e:
        logger.error(f"Error generating QR code: {e}")
        return ""

def apply_ar_overlay(base_image_path: Path, overlay_path: Path, output_path: Path) -> Path:
    """Apply AR overlay to captured photo"""
    try:
        if not overlay_path.exists():
            logger.warning(f"Overlay file not found: {overlay_path}")
            # Just copy the original if no overlay
            shutil.copy2(base_image_path, output_path)
            return output_path
        
        base_img = cv2.imread(str(base_image_path), cv2.IMREAD_UNCHANGED)
        overlay_img = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
        
        if base_img is None:
            raise ARPhotoError(f"Cannot read base image: {base_image_path}")
        if overlay_img is None:
            logger.warning(f"Cannot read overlay image: {overlay_path}")
            shutil.copy2(base_image_path, output_path)
            return output_path
        
        # Resize overlay to match base image
        overlay_img = cv2.resize(overlay_img, (base_img.shape[1], base_img.shape[0]))
        
        # Apply overlay with alpha blending if available
        if overlay_img.shape[2] == 4:  # Has alpha channel
            alpha_mask = overlay_img[:, :, 3] / 255.0
            alpha_mask = np.stack([alpha_mask] * 3, axis=2)
            
            base_img_rgb = base_img[:, :, :3] if base_img.shape[2] >= 3 else base_img
            overlay_img_rgb = overlay_img[:, :, :3]
            
            result = (1 - alpha_mask) * base_img_rgb + alpha_mask * overlay_img_rgb
            result = result.astype(np.uint8)
        else:
            # Simple overlay without alpha
            result = cv2.addWeighted(base_img, 0.7, overlay_img, 0.3, 0)
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), result)
        
        if not success:
            raise ARPhotoError("Failed to save AR overlay result")
        
        logger.info(f"AR overlay applied: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"AR overlay error: {e}")
        # Fallback: copy original file
        shutil.copy2(base_image_path, output_path)
        return output_path

def create_ar_directories():
    """Create necessary directories for AR Photo functionality"""
    directories = [
        ARConfig.AR_ASSETS_DIR,
        ARConfig.COUNTDOWN_DIR,
        ARConfig.THUMBNAIL_DIR,
        ARConfig.AR_RESULTS_DIR,
        ARConfig.UPLOAD_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"AR directory created/verified: {directory}")

def generate_countdown_videos():
    """Generate countdown videos using script"""
    try:
        from countdown_generator import CountdownGenerator
        generator = CountdownGenerator(ARConfig.COUNTDOWN_DIR)
        countdown_files = generator.generate_all_countdowns()
        logger.info("Countdown videos generated successfully")
        return countdown_files
    except ImportError:
        logger.warning("Countdown generator not available")
        return []
    except Exception as e:
        logger.error(f"Error generating countdown videos: {e}")
        return []

# Authentication dependency (will be injected from main.py)
async def get_current_user_ar(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    This will be overridden by the main app when including the router
    For now, we'll import auth service directly
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        # Import here to avoid circular import
        from main import auth_service
        token = credentials.credentials
        return auth_service.get_user_by_token(token)
    except ImportError:
        # Fallback for testing
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication service not available"
        )

# ===== AR PHOTO ROUTES =====

@router.get("/characters")
async def list_ar_characters():
    """List available AR character thumbnails and webm files"""
    try:
        characters = []
        
        # Ensure directories exist
        ARConfig.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
        ARConfig.AR_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        
        if ARConfig.THUMBNAIL_DIR.exists():
            for thumbnail_file in ARConfig.THUMBNAIL_DIR.iterdir():
                if thumbnail_file.is_file() and thumbnail_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    character_name = thumbnail_file.stem  # filename without extension
                    
                    # Look for corresponding webm file
                    webm_file = ARConfig.AR_ASSETS_DIR / f"{character_name}.webm"
                    
                    character_data = {
                        "name": character_name,
                        "thumbnail": f"/static/ar_assets/thumbnail/{thumbnail_file.name}",
                        "has_animation": webm_file.exists(),
                        "animation_url": f"/static/ar_assets/{character_name}.webm" if webm_file.exists() else None
                    }
                    
                    characters.append(character_data)
        
        # If no characters exist, create sample data
        if not characters:
            sample_characters = [
                {
                    "name": "boy",
                    "thumbnail": "/static/ar_assets/thumbnail/boy.png",
                    "has_animation": True,
                    "animation_url": "/static/ar_assets/boy.webm"
                },
                {
                    "name": "girl", 
                    "thumbnail": "/static/ar_assets/thumbnail/ghost.png",
                    "has_animation": True,
                    "animation_url": "/static/ar_assets/ghost.webm"
                },
                {
                    "name": "superhero",
                    "thumbnail": "/static/ar_assets/thumbnail/superman.png", 
                    "has_animation": True,
                    "animation_url": "/static/ar_assets/superman.webm"
                }
            ]
            characters = sample_characters
        
        return JSONResponse({
            "success": True,
            "characters": characters,
            "count": len(characters)
        })
    
    except Exception as e:
        logger.error(f"Error listing AR characters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AR characters"
        )

@router.get("/assets/countdown")
async def list_countdown_assets():
    """List available countdown video files"""
    try:
        countdown_files = []
        ARConfig.COUNTDOWN_DIR.mkdir(parents=True, exist_ok=True)
        
        if ARConfig.COUNTDOWN_DIR.exists():
            for file_path in ARConfig.COUNTDOWN_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.webm', '.mov']:
                    countdown_files.append({
                        "name": file_path.name,
                        "path": f"/static/ar_assets/countdown/{file_path.name}",
                        "duration": 1  # You can add actual duration detection here
                    })
        
        # If no countdown files, create sample data
        if not countdown_files:
            sample_countdowns = [
                {"name": "countdown_5.mp4", "path": "/static/ar_assets/countdown/countdown_5.mp4", "duration": 1},
                {"name": "countdown_4.mp4", "path": "/static/ar_assets/countdown/countdown_4.mp4", "duration": 1},
                {"name": "countdown_3.mp4", "path": "/static/ar_assets/countdown/countdown_3.mp4", "duration": 1},
                {"name": "countdown_2.mp4", "path": "/static/ar_assets/countdown/countdown_2.mp4", "duration": 1},
                {"name": "countdown_1.mp4", "path": "/static/ar_assets/countdown/countdown_1.mp4", "duration": 1},
                {"name": "countdown_go.mp4", "path": "/static/ar_assets/countdown/countdown_go.mp4", "duration": 1}
            ]
            countdown_files = sample_countdowns
        
        return JSONResponse({
            "success": True,
            "countdown_assets": countdown_files,
            "total_duration": len(countdown_files),
            "count": len(countdown_files)
        })
    
    except Exception as e:
        logger.error(f"Error listing countdown assets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get countdown assets"
        )

@router.post("/photo")
async def create_ar_photo(
    photo: UploadFile = File(...),
    overlay_name: str = Form("frame1.png"),
    current_user = Depends(get_current_user_ar)
):
    """Process captured photo with AR overlay"""
    temp_files = []
    
    try:
        # Validate uploaded photo
        validate_file(photo)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Ensure directories exist
        ARConfig.AR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ARConfig.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded photo
        photo_filename = generate_unique_filename(photo.filename, "ar_capture_")
        photo_path = ARConfig.UPLOAD_DIR / photo_filename
        await save_uploaded_file(photo, photo_path)
        temp_files.append(photo_path)
        
        # Define overlay path
        overlay_path = ARConfig.FRAME_DIR / overlay_name
        
        # Create result filename
        result_filename = f"ar_photo_{timestamp}_{unique_id}.png"
        result_path = ARConfig.AR_RESULTS_DIR / result_filename
        
        # Apply AR overlay
        logger.info(f"Applying AR overlay: {photo_path} + {overlay_path}")
        final_result_path = apply_ar_overlay(photo_path, overlay_path, result_path)
        
        # Generate download URL and QR code
        download_url = f"/static/ar_results/{result_filename}"
        full_download_url = f"{ARConfig.DOMAIN_URL}{download_url}"
        qr_code_data = generate_qr_code(full_download_url)
        
        # Save to database
        try:
            from main import auth_service
            with auth_service.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT INTO face_swap_history (user_id, template_name, result_filename)
                    VALUES (?, ?, ?)
                """, (current_user["id"], f"AR_OVERLAY_{overlay_name}", result_filename))
                conn.commit()
        except ImportError:
            logger.warning("Database service not available")
        
        response_data = {
            "success": True,
            "message": "AR photo berhasil dibuat",
            "data": {
                "result_url": download_url,
                "result_filename": result_filename,
                "qr_code": qr_code_data,
                "download_url": full_download_url,
                "overlay_used": overlay_name,
                "processing_time": datetime.now().isoformat(),
                "file_size": result_path.stat().st_size if result_path.exists() else 0
            }
        }
        
        logger.info(f"AR photo created successfully: {final_result_path}")
        return JSONResponse(response_data)
    
    except Exception as e:
        logger.error(f"Unexpected error in AR photo creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan dalam pembuatan AR photo"
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

@router.get("/overlays")
async def list_ar_overlays():
    """List available AR overlay frames"""
    try:
        overlays = []
        ARConfig.FRAME_DIR.mkdir(parents=True, exist_ok=True)
        
        if ARConfig.FRAME_DIR.exists():
            for file_path in ARConfig.FRAME_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ARConfig.ALLOWED_EXTENSIONS:
                    overlays.append({
                        "name": file_path.name,
                        "path": f"/static/images/{file_path.name}",
                        "preview": f"/static/images/{file_path.name}"
                    })
        
        # Default frame if no overlays exist
        if not overlays:
            overlays = [
                {"name": "frame1.png", "path": "/static/images/frame1.png", "preview": "/static/images/frame1.png"}
            ]
        
        return JSONResponse({
            "success": True,
            "overlays": overlays,
            "count": len(overlays)
        })
    
    except Exception as e:
        logger.error(f"Error listing AR overlays: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AR overlays"
        )

@router.get("/test")
async def ar_test():
    """Test AR endpoints"""
    return {
        "success": True,
        "message": "AR Photo module working!",
        "endpoints": [
            "/api/ar/characters",
            "/api/ar/photo",
            "/api/ar/overlays",
            "/api/ar/assets/countdown",
            "/api/ar/test"
        ],
        "status": "OK"
    }

# Initialize AR Photo module
def init_ar_photo():
    """Initialize AR Photo module"""
    try:
        create_ar_directories()
        
        # Generate countdown videos if they don't exist
        countdown_dir = ARConfig.COUNTDOWN_DIR
        existing_countdowns = list(countdown_dir.glob("countdown_*.mp4"))
        
        if len(existing_countdowns) < 6:  # Should have 5 numbers + GO
            logger.info("Generating countdown videos...")
            generate_countdown_videos()
        
        logger.info("AR Photo module initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AR Photo module: {e}")
        raise