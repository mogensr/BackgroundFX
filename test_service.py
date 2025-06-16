"""
Test script for BackgroundFX microservice
"""
import os
import sys
import time
import cv2
import requests
import base64
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BackgroundFX-Test")

def test_service_health():
    """Test the health endpoint of the BackgroundFX service"""
    logger.info("Testing service health...")
    
    try:
        response = requests.get("http://localhost:5000/health", timeout=5.0)
        response.raise_for_status()
        health_info = response.json()
        
        logger.info(f"Service health: {health_info}")
        logger.info(f"GPU available: {health_info.get('gpu_available', False)}")
        
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def test_background_replacement():
    """Test background replacement with a sample image"""
    logger.info("Testing background replacement...")
    
    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    
    # Use sample image - a simple solid color square if no image exists
    test_image_path = "test_data/test_image.jpg"
    if not os.path.exists(test_image_path):
        logger.info("Creating sample test image...")
        # Create a simple test image - red square on white background
        img = cv2.imread("test_data/test_image.jpg") if os.path.exists("test_data/test_image.jpg") else None
        if img is None:
            img = np.ones((300, 300, 3), dtype=np.uint8) * 255  # White background
            img[50:250, 50:250] = [0, 0, 200]  # Red square
            cv2.imwrite(test_image_path, img)
    
    # Create a sample background
    bg_image_path = "test_data/test_background.jpg"
    if not os.path.exists(bg_image_path):
        logger.info("Creating sample background image...")
        # Create a simple background - gradient
        bg = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(300):
            bg[:, i] = [i // 2, 100, 255 - i // 2]  # Gradient background
        cv2.imwrite(bg_image_path, bg)
    
    # Read and encode the test image
    with open(test_image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Read and encode the background
    with open(bg_image_path, "rb") as f:
        bg_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Prepare request data
    request_data = {
        "image_base64": image_base64,
        "background_base64": bg_base64,
        "quality": "medium"
    }
    
    try:
        logger.info("Sending background replacement request...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:5000/replace-background",
            json=request_data,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("success", False):
            processing_time = time.time() - start_time
            logger.info(f"Background replacement succeeded in {processing_time:.2f} seconds")
            
            # Save the result image
            result_image_path = "test_data/result_image.jpg"
            result_image_data = base64.b64decode(result["image_base64"])
            with open(result_image_path, "wb") as f:
                f.write(result_image_data)
                
            logger.info(f"Result saved to {result_image_path}")
            
            return True
        else:
            logger.error(f"Background replacement failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Background replacement request failed: {e}")
        return False

def test_file_upload():
    """Test background replacement with file upload"""
    logger.info("Testing file upload...")
    
    # Use sample image from previous test
    test_image_path = "test_data/test_image.jpg"
    if not os.path.exists(test_image_path):
        logger.error(f"Test image not found: {test_image_path}")
        return False
    
    # Prepare form data
    files = {
        "file": open(test_image_path, "rb")
    }
    
    form_data = {
        "quality": "high"
    }
    
    try:
        logger.info("Sending file upload request...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:5000/upload-and-replace",
            files=files,
            data=form_data,
            timeout=30.0
        )
        response.raise_for_status()
        
        processing_time = time.time() - start_time
        logger.info(f"File upload and processing succeeded in {processing_time:.2f} seconds")
        
        # Save the result image
        result_image_path = "test_data/result_upload.jpg"
        with open(result_image_path, "wb") as f:
            f.write(response.content)
            
        logger.info(f"Result saved to {result_image_path}")
        
        return True
            
    except Exception as e:
        logger.error(f"File upload request failed: {e}")
        return False
    finally:
        # Make sure to close the file
        files["file"].close()

def main():
    """Run all tests"""
    logger.info("Starting BackgroundFX service tests")
    
    # Test health endpoint
    health_ok = test_service_health()
    
    if not health_ok:
        logger.error("Health check failed, skipping other tests")
        return 1
    
    # Test background replacement
    bg_replace_ok = test_background_replacement()
    
    # Test file upload
    upload_ok = test_file_upload()
    
    # Report results
    logger.info("\n=== Test Results ===")
    logger.info(f"Service Health: {'✅ PASS' if health_ok else '❌ FAIL'}")
    logger.info(f"Background Replacement: {'✅ PASS' if bg_replace_ok else '❌ FAIL'}")
    logger.info(f"File Upload: {'✅ PASS' if upload_ok else '❌ FAIL'}")
    
    if health_ok and bg_replace_ok and upload_ok:
        logger.info("All tests passed! BackgroundFX service is working properly.")
        return 0
    else:
        logger.error("Some tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        logger.error("Required dependencies missing. Please install the requirements first.")
        sys.exit(1)
    
    sys.exit(main())
