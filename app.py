import gradio as gr
import os
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import tempfile
import shutil
import logging
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI Model Availability & Loading ---
SAM2_AVAILABLE = False
SAM2_PREDICTOR = None

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_PREDICTOR = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    SAM2_AVAILABLE = True
    logger.info("✅ SAM2 (Large Model) loaded successfully")
except ImportError:
    logger.warning("⚠️ SAM2 not available. Please ensure it's installed via requirements.txt.")
except Exception as e:
    logger.error(f"🚨 Error loading SAM2 model: {e}")

# --- Background Options ---
def get_background_options():
    return {
        "Beach Sunset": "https://images.pexels.com/photos/237272/pexels-photo-237272.jpeg",
        "Modern Office": "https://images.pexels.com/photos/1181244/pexels-photo-1181244.jpeg",
        "Cozy Cafe": "https://images.pexels.com/photos/262047/pexels-photo-262047.jpeg",
        "Mountain Landscape": "https://images.pexels.com/photos/572897/pexels-photo-572897.jpeg",
        "Custom": None,
        "Solid Color": None
    }

# --- Core Processing Logic ---
def segment_person_sam2(image_np, predictor):
    if not SAM2_AVAILABLE or predictor is None:
        raise gr.Error("SAM2 model is not available. Cannot perform segmentation.")
    
    input_image = Image.fromarray(image_np)
    everything_results = predictor.predict_everything(input_image, verbose=False)
    
    if not everything_results:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    # Heuristic to find the person: often the largest or most central mask
    # For simplicity, we'll merge all masks. A better approach would be to classify them.
    # This part can be improved with more advanced logic (e.g., text prompts if model supports it)
    full_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    for mask_result in everything_results:
        full_mask = np.logical_or(full_mask, mask_result.mask)
        
    return full_mask.astype(np.uint8) * 255

def process_video(video_path, background_choice, custom_bg_image, solid_color, progress=gr.Progress(track_tqdm=True)):
    if video_path is None:
        raise gr.Error("Please upload a video first.")

    # --- Background Setup ---
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if background_choice == "Custom":
        if custom_bg_image is None: raise gr.Error("Please upload a custom background image.")
        background_image = np.array(custom_bg_image)
    elif background_choice == "Solid Color":
        if solid_color is None: raise gr.Error("Please choose a color.")
        color_rgb = tuple(int(solid_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        background_image = np.full((height, width, 3), color_rgb, dtype=np.uint8)
    else:
        background_url = get_background_options()[background_choice]
        background_image = np.array(Image.open(BytesIO(requests.get(background_url).content)).convert("RGB"))

    background_resized = cv2.resize(background_image, (width, height))

    # --- Video Processing ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get person mask using SAM2
        person_mask = segment_person_sam2(frame_rgb, SAM2_PREDICTOR)
        person_mask_inv = cv2.bitwise_not(person_mask)

        # Composite the new background
        background_part = cv2.bitwise_and(background_resized, background_resized, mask=person_mask_inv)
        foreground_part = cv2.bitwise_and(frame, frame, mask=person_mask)
        combined_frame = cv2.add(background_part, foreground_part)
        
        out.write(combined_frame)

    cap.release()
    out.release()
    
    return output_path

# --- Gradio UI ---
def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="BackgroundFX") as demo:
        gr.Markdown("""
        # 🎬 BackgroundFX: Video Background Replacement
        Upload a video, choose a new background, and let the AI do the rest.
        **Note:** Processing can be slow, especially for long videos. Requires a GPU for reasonable speed.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Your Video")
                background_choice = gr.Radio(
                    choices=list(get_background_options().keys()), 
                    label="Choose a Background", 
                    value="Beach Sunset"
                )
                with gr.Accordion("Custom Background Options", open=False):
                    custom_bg_image = gr.Image(label="Upload Custom Image", type="pil")
                    solid_color = gr.ColorPicker(label="Choose Solid Color")
                
                process_button = gr.Button("✨ Process Video", variant="primary")

            with gr.Column(scale=2):
                video_output = gr.Video(label="Processed Video")

        process_button.click(
            fn=process_video,
            inputs=[video_input, background_choice, custom_bg_image, solid_color],
            outputs=video_output
        )

    return demo

if __name__ == "__main__":
    app = build_ui()
    app.launch(debug=True)
