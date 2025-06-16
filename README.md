# BackgroundFX Microservice

A dedicated microservice for background replacement in images and videos, part of the heavylift microservices collection.

## Features

- Image background replacement using AI-based segmentation
- High-quality foreground extraction with RobustVideoMatting
- Fallback to OpenCV-based segmentation when GPU is not available
- Various quality levels to balance performance and results
- REST API for easy integration with other applications
- libraryFX integration for notifications and configuration

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /replace-background` - Replace background in base64-encoded image
- `POST /upload-and-replace` - Replace background in uploaded image file
- `POST /video/replace-background` - Replace background in video (async processing)
- `POST /backgrounds/upload` - Upload a new background image
- `GET /backgrounds` - List available backgrounds

## Setup and Installation

1. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:

```
pip install -r requirements.txt
```

3. Configure the service by creating a `.env` file (see `.env.example`)

4. Run the service:

```
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

## Integration with MyAvatar

BackgroundFX is designed to be used as a standalone microservice for background replacement in MyAvatar. To integrate with MyAvatar:

1. Ensure BackgroundFX is running on a specified port
2. Update the MyAvatar configuration to point to the BackgroundFX service URL
3. Use the BackgroundFX client in MyAvatar to communicate with the service

## API Usage Examples

### Replace background in an image (base64)

```python
import requests
import base64
import json

# Load image
with open("person.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode("utf-8")

# Request body
data = {
    "image_base64": img_data,
    "background_id": 1,  # Or use background_base64 for a custom background
    "quality": "high"
}

# Make request
response = requests.post(
    "http://localhost:5000/replace-background",
    json=data
)

# Save result
if response.status_code == 200 and response.json().get("success"):
    result_img = base64.b64decode(response.json()["image_base64"])
    with open("result.jpg", "wb") as f:
        f.write(result_img)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Port to run the service on | `5000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENABLE_GPU` | Whether to use GPU if available | `TRUE` |

## Performance Considerations

- The RVM segmentation model performs best on GPU
- For high-volume processing, consider scaling horizontally
- Video processing is resource-intensive and handled asynchronously
