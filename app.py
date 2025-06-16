"""
BackgroundFX Microservice
A dedicated service for background replacement in images and videos
Part of the heavylift microservices collection
"""
import os
import cv2
import base64
import logging
import time
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import uvicorn
from dotenv import load_dotenv
import sys

# Setup paths for importing from libraryFX
LIBRARY_FX_PATH = os.path.join(Path.home(), "Projects", "Python", "libraryFX")
if os.path.exists(LIBRARY_FX_PATH):
    sys.path.insert(0, LIBRARY_FX_PATH)
    
    # Try to import from libraryFX
    try:
        from libraryFX.notifications.alerts import send_alert
        from libraryFX.core.config import load_config
        USING_LIBRARY_FX = True
    except ImportError:
        USING_LIBRARY_FX = False
        
        # Fallback dummy functions
        def send_alert(title, message, severity="info"):
            logging.warning(f"[ALERT-{severity.upper()}] {title}: {message}")
            
        def load_config(key, default=None):
            return os.environ.get(key, default)
else:
    USING_LIBRARY_FX = False
    
    # Fallback dummy functions
    def send_alert(title, message, severity="info"):
        logging.warning(f"[ALERT-{severity.upper()}] {title}: {message}")
        
    def load_config(key, default=None):
        return os.environ.get(key, default)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BackgroundFX")

# Import the background replacement module
from segmentation import BackgroundReplacer, RVMSegmenter, FallbackSegmenter

# Create FastAPI app
app = FastAPI(
    title="BackgroundFX API",
    description="Microservice for image and video background replacement",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize background replacer
background_replacer = None

# Create required directories
os.makedirs("temp", exist_ok=True)
os.makedirs("backgrounds", exist_ok=True)

# API Models
class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool = False

class ReplaceBackgroundRequest(BaseModel):
    image_base64: str
    background_id: Optional[int] = None
    background_base64: Optional[str] = None
    quality: str = "medium"

class ReplaceBackgroundResponse(BaseModel):
    success: bool
    message: str 
    image_base64: Optional[str] = None
    processing_time: Optional[float] = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is running and whether GPU is available"""
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
        
    return {
        "status": "ok",
        "version": "1.0.0",
        "gpu_available": gpu_available
    }

# Background replacement endpoint for images (base64)
@app.post("/replace-background", response_model=ReplaceBackgroundResponse)
async def replace_background(request: ReplaceBackgroundRequest):
    """Replace background in an image using base64 encoding"""
    global background_replacer
    
    # Initialize background replacer if not already done
    if background_replacer is None:
        try:
            background_replacer = BackgroundReplacer(enable_timing=True)
            logger.info("Background replacer initialized successfully")
        except Exception as e:
            send_alert(
                title="BackgroundFX Initialization Failed",
                message=f"Failed to initialize background replacer: {str(e)}",
                severity="error"
            )
            logger.error(f"Failed to initialize background replacer: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Failed to initialize background replacer: {str(e)}",
                    "image_base64": None,
                    "processing_time": None
                }
            )
    
    start_time = time.time()
    
    try:
        # Decode input image
        try:
            image_data = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image data")
        except Exception as e:
            logger.error(f"Error decoding input image: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"Invalid input image: {str(e)}",
                    "image_base64": None,
                    "processing_time": None
                }
            )
        
        # Get background image
        if request.background_base64:
            try:
                bg_data = base64.b64decode(request.background_base64)
                bg_nparr = np.frombuffer(bg_data, np.uint8)
                bg_image = cv2.imdecode(bg_nparr, cv2.IMREAD_COLOR)
                if bg_image is None:
                    raise ValueError("Invalid background image data")
            except Exception as e:
                logger.error(f"Error decoding background image: {e}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": f"Invalid background image: {str(e)}",
                        "image_base64": None,
                        "processing_time": None
                    }
                )
        elif request.background_id is not None:
            # In a real implementation, you would fetch the background image from a database or storage
            # For now, we'll use a default background if available
            bg_filename = f"background_{request.background_id}.jpg"
            bg_path = os.path.join("backgrounds", bg_filename)
            
            if not os.path.exists(bg_path):
                # Use a solid color background as fallback
                bg_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 100
                bg_image[:, :, 0] = 50  # Blue channel
                bg_image[:, :, 1] = 100  # Green channel
                bg_image[:, :, 2] = 150  # Red channel
                logger.warning(f"Background ID {request.background_id} not found, using default")
            else:
                bg_image = cv2.imread(bg_path)
                # Resize to match input image size
                bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
        else:
            # Use a default blue background
            bg_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 100
            bg_image[:, :, 0] = 200  # Blue channel
            bg_image[:, :, 1] = 100  # Green channel
            bg_image[:, :, 2] = 100  # Red channel
        
        # Process the image with background replacement
        try:
            result = background_replacer.replace_background(image, bg_image, quality=request.quality)
        except Exception as e:
            logger.error(f"Error during background replacement: {e}")
            send_alert(
                title="BackgroundFX Processing Error",
                message=f"Error during background replacement: {str(e)}",
                severity="error"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Error processing image: {str(e)}",
                    "image_base64": None,
                    "processing_time": None
                }
            )
        
        # Encode output image as base64
        _, buffer = cv2.imencode(".jpg", result)
        output_base64 = base64.b64encode(buffer).decode("utf-8")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Background replacement completed in {processing_time:.2f} seconds")
        
        # Return the processed image
        return {
            "success": True,
            "message": "Background replacement completed successfully",
            "image_base64": output_base64,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Unexpected error during background replacement: {e}")
        send_alert(
            title="BackgroundFX Unexpected Error",
            message=f"Unexpected error: {str(e)}",
            severity="error"
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"An unexpected error occurred: {str(e)}",
                "image_base64": None,
                "processing_time": None
            }
        )

# File upload endpoint for background replacement
@app.post("/upload-and-replace")
async def upload_and_replace(
    file: UploadFile = File(...),
    background_id: int = Form(None),
    quality: str = Form("medium")
):
    """Replace background in an uploaded image file"""
    global background_replacer
    
    # Initialize background replacer if not already done
    if background_replacer is None:
        try:
            background_replacer = BackgroundReplacer(enable_timing=True)
            logger.info("Background replacer initialized successfully")
        except Exception as e:
            send_alert(
                title="BackgroundFX Initialization Failed",
                message=f"Failed to initialize background replacer: {str(e)}",
                severity="error"
            )
            logger.error(f"Failed to initialize background replacer: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize background replacer: {str(e)}")
    
    start_time = time.time()
    
    try:
        # Read and decode input image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get background image
        if background_id is not None:
            bg_filename = f"background_{background_id}.jpg"
            bg_path = os.path.join("backgrounds", bg_filename)
            
            if not os.path.exists(bg_path):
                # Use a solid color background as fallback
                bg_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 100
                bg_image[:, :, 0] = 50  # Blue channel
                bg_image[:, :, 1] = 100  # Green channel
                bg_image[:, :, 2] = 150  # Red channel
                logger.warning(f"Background ID {background_id} not found, using default")
            else:
                bg_image = cv2.imread(bg_path)
                # Resize to match input image size
                bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
        else:
            # Use a default blue background
            bg_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 100
            bg_image[:, :, 0] = 200  # Blue channel
            bg_image[:, :, 1] = 100  # Green channel
            bg_image[:, :, 2] = 100  # Red channel
        
        # Process the image
        result = background_replacer.replace_background(image, bg_image, quality=quality)
        
        # Create an in-memory bytes buffer for the output image
        is_success, buffer = cv2.imencode(".jpg", result)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode output image")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Background replacement completed in {processing_time:.2f} seconds")
        
        # Return the processed image as a streaming response
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "X-Processing-Time": str(processing_time),
                "Content-Disposition": f"attachment; filename=bg_replaced_{file.filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        send_alert(
            title="BackgroundFX Processing Error",
            message=f"Error processing uploaded image: {str(e)}",
            severity="error"
        )
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# For processing videos
@app.post("/video/replace-background")
async def replace_video_background(
    video_file: UploadFile = File(...),
    background_id: int = Form(...),
    quality: str = Form("medium")
):
    """
    Replace background in a video file
    WARNING: This is a resource-intensive operation
    """
    # In a real implementation, this would likely be handled as an async task
    # For now, we'll return a simple message
    return JSONResponse(content={
        "success": True,
        "message": "Video background replacement submitted",
        "job_id": f"video_{int(time.time())}",
        "estimated_time": "60 seconds"
    })

# Simple upload endpoint for backgrounds
@app.post("/backgrounds/upload")
async def upload_background(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form("")
):
    """Upload a new background image"""
    try:
        # Generate a unique ID for the background
        background_id = int(time.time())
        filename = f"background_{background_id}.jpg"
        file_path = os.path.join("backgrounds", filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return JSONResponse(content={
            "success": True,
            "background_id": background_id,
            "name": name,
            "description": description,
            "file_path": file_path
        })
    except Exception as e:
        logger.error(f"Error uploading background: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error uploading background: {str(e)}"
            }
        )

# List available backgrounds
@app.get("/backgrounds")
async def list_backgrounds():
    """List all available backgrounds"""
    try:
        backgrounds = []
        backgrounds_dir = "backgrounds"
        
        if os.path.exists(backgrounds_dir):
            for filename in os.listdir(backgrounds_dir):
                if filename.startswith("background_") and (filename.endswith(".jpg") or filename.endswith(".png")):
                    try:
                        bg_id = int(filename.split("_")[1].split(".")[0])
                        backgrounds.append({
                            "id": bg_id,
                            "name": f"Background {bg_id}",
                            "path": os.path.join(backgrounds_dir, filename)
                        })
                    except (IndexError, ValueError):
                        continue
        
        return JSONResponse(content={
            "success": True,
            "backgrounds": backgrounds
        })
    except Exception as e:
        logger.error(f"Error listing backgrounds: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error listing backgrounds: {str(e)}"
            }
        )

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("BackgroundFX microservice starting up")
    
    if USING_LIBRARY_FX:
        logger.info("Using libraryFX for notifications and configuration")
    else:
        logger.warning("libraryFX not available, using local fallbacks")
    
    # Check if we can initialize the background replacer
    try:
        global background_replacer
        background_replacer = BackgroundReplacer(enable_timing=True)
        logger.info("Background replacer initialized successfully on startup")
    except Exception as e:
        logger.error(f"Failed to initialize background replacer on startup: {e}")
        # We'll initialize it on first request instead

    # Report successful startup
    send_alert(
        title="BackgroundFX Startup",
        message="BackgroundFX microservice started successfully",
        severity="info"
    )

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the FastAPI application
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
