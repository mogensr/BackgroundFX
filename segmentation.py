"""
Background replacement module for BackgroundFX microservice
Uses segmentation to separate subject from background and replace it
"""
import cv2
import numpy as np
import logging
import time
from typing import Optional, List, Dict, Tuple
import torch

# Configure logging
logger = logging.getLogger(__name__)

class BackgroundReplacer:
    """
    Background replacement utility that uses RVM segmentation
    to replace video backgrounds with user-selected images
    """
    
    def __init__(self, segmentation_model=None, enable_timing=False):
        """
        Initialize the background replacer
        
        Args:
            segmentation_model: Optional segmentation model (will load RVM if None)
            enable_timing: Whether to enable performance timing
        """
        # Initialize segmentation model
        if segmentation_model is None:
            try:
                logger.info("Initializing RobustVideoMatting model...")
                # Try to import and init RVM model
                try:
                    import torch
                    self.segment = RVMSegmenter()
                except (ImportError, RuntimeError) as e:
                    logger.warning(f"Could not load RVM model: {e}")
                    logger.warning("Using fallback OpenCV segmentation")
                    self.segment = FallbackSegmenter()
            except Exception as e:
                logger.error(f"Failed to initialize any segmentation model: {e}")
                raise RuntimeError("Could not initialize segmentation model")
        else:
            self.segment = segmentation_model
        
        # Performance tracking
        self.enable_timing = enable_timing
        self.frame_count = 0
        self.total_times = {}
    
    def _log_timing(self, step_name: str, duration: float):
        """Log performance timing data"""
        if not self.enable_timing:
            return
        if step_name not in self.total_times:
            self.total_times[step_name] = []
        self.total_times[step_name].append(duration)
        if self.frame_count % 50 == 0:
            logger.info(f"Frame {self.frame_count} - {step_name}: {duration*1000:.1f}ms")
    
    def get_timing_report(self) -> str:
        """Get performance timing report"""
        if not self.total_times:
            return "No timing data available"
        
        report = "\n=== BACKGROUND REPLACER TIMING REPORT ===\n"
        total_avg = 0
        
        for step, times in self.total_times.items():
            avg_time = np.mean(times)
            total_avg += avg_time
            max_time = np.max(times)
            min_time = np.min(times)
            report += f"{step:15s}: {avg_time*1000:6.1f}ms avg (min: {min_time*1000:5.1f}ms, max: {max_time*1000:5.1f}ms)\n"
        
        report += f"{'TOTAL':15s}: {total_avg*1000:6.1f}ms avg ({1/total_avg:4.1f} FPS)\n"
        report += f"Frames processed: {self.frame_count}\n"
        return report
    
    def _feather_mask(self, mask: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        Apply feathering to mask edges for smoother compositing
        
        Args:
            mask: Binary mask
            radius: Feathering radius
            
        Returns:
            Feathered alpha mask as float32 array (0.0-1.0)
        """
        alpha = mask.astype(np.float32) / 255.0
        # Apply gaussian blur for anti-aliasing and feathering
        alpha = cv2.GaussianBlur(alpha, (radius*2+1, radius*2+1), radius/2)
        return np.clip(alpha, 0.0, 1.0)
    
    def _sharpen_edges(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Sharpen edges in the image
        
        Args:
            img: Input image
            strength: Sharpening strength
            
        Returns:
            Sharpened image
        """
        # Create a slight blur
        blurred = cv2.GaussianBlur(img, (3, 3), 1.0)
        
        # Calculate difference (edges)
        diff = img.astype(np.float32) - blurred.astype(np.float32)
        
        # Add sharpening
        sharpened = img.astype(np.float32) + diff * strength
        
        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened
    
    def replace_background(self, frame: np.ndarray, bg_img: np.ndarray, 
                           quality: str = "medium") -> np.ndarray:
        """
        Replace the background of a frame with the provided image
        
        Args:
            frame: Input video frame (BGR format)
            bg_img: Background image
            quality: Quality level ("low", "medium", "high")
            
        Returns:
            Frame with replaced background
        """
        start = time.time()
        
        # Track total frames
        self.frame_count += 1
        
        # Resize background to match frame size if needed
        if bg_img.shape[:2] != frame.shape[:2]:
            bg_img = cv2.resize(bg_img, (frame.shape[1], frame.shape[0]))
        
        # Set quality parameters
        if quality.lower() == "low":
            segmentation_downscale = 0.25
            feather_radius = 1
            sharpen_strength = 0.0
        elif quality.lower() == "medium":
            segmentation_downscale = 0.5
            feather_radius = 3
            sharpen_strength = 0.5
        else:  # high
            segmentation_downscale = 0.8
            feather_radius = 5
            sharpen_strength = 0.8
        
        # 1. Get segmentation mask
        t0 = time.time()
        mask = self.segment.segment_frame(frame, downsample_ratio=segmentation_downscale)
        self._log_timing("segmentation", time.time() - t0)
        
        # 2. Apply feathering to mask edges
        t0 = time.time()
        alpha = self._feather_mask(mask, radius=feather_radius)
        self._log_timing("feathering", time.time() - t0)
        
        # 3. Prepare alpha for compositing (convert to 3-channel)
        t0 = time.time()
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
        
        # 4. Apply compositing formula: foreground * alpha + background * (1 - alpha)
        result = (frame.astype(np.float32) * alpha_3ch + 
                  bg_img.astype(np.float32) * (1.0 - alpha_3ch))
        result = np.clip(result, 0, 255).astype(np.uint8)
        self._log_timing("compositing", time.time() - t0)
        
        # 5. Optional: Sharpen edges if quality is medium or high
        if sharpen_strength > 0:
            t0 = time.time()
            result = self._sharpen_edges(result, strength=sharpen_strength)
            self._log_timing("sharpening", time.time() - t0)
        
        # Track total time
        self._log_timing("total", time.time() - start)
        
        return result


class RVMSegmenter:
    """
    Segmenter using RobustVideoMatting for high-quality segmentation
    """
    
    def __init__(self, model_name: str = 'mobilenetv3'):
        """Initialize RobustVideoMatting model"""
        
        # Check for GPU availability
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.device = torch.device('cpu')
            logger.warning("Using CPU - GPU not available")
        
        # Load model from torch hub
        logger.info("Loading RobustVideoMatting model...")
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
        
        # For recurrent inference
        self.rec = [None] * 4
        
    @torch.no_grad()
    def segment_frame(self, frame: np.ndarray, downsample_ratio: float = 0.5) -> np.ndarray:
        """
        Segment a single frame using RVM
        
        Args:
            frame: Input frame (BGR format)
            downsample_ratio: Ratio to downsample for processing
            
        Returns:
            Segmentation mask as numpy array (H, W) with values 0-255
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        src = torch.from_numpy(frame_rgb).float() / 255.0
        src = src.permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        src = src.to(self.device)
        
        # Forward pass with downsample ratio
        fgr, pha, *self.rec = self.model(src, *self.rec, downsample_ratio=downsample_ratio)
        
        # Convert mask to numpy
        mask = pha[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        
        # Resize mask to original frame size
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        return mask


class FallbackSegmenter:
    """
    Fallback segmenter using OpenCV for when RVM is not available
    Uses simpler but less accurate methods
    """
    
    def __init__(self):
        """Initialize the fallback segmenter"""
        logger.warning("Using fallback OpenCV segmentation - results will be lower quality")
        
    def segment_frame(self, frame: np.ndarray, downsample_ratio: float = 0.5) -> np.ndarray:
        """
        Segment a single frame using OpenCV methods
        
        Args:
            frame: Input frame (BGR format)
            downsample_ratio: Ignored in fallback mode
            
        Returns:
            Segmentation mask as numpy array (H, W) with values 0-255
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Use adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 4. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Create mask from largest contours
        mask = np.zeros_like(gray)
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Take top 3 contours
        for i in range(min(3, len(contours))):
            cv2.drawContours(mask, [contours[i]], -1, 255, -1)
        
        # 6. Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
