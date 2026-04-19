# PPE Multi-Model Detection System

A Streamlit-based web application for detecting Personal Protective Equipment (PPE) in images using multiple state-of-the-art deep learning models. The system supports both image uploads and live camera capture, providing real-time inference with visual annotations.

## Features

- **Multi-model inference**: Choose from three pre-trained models with different trade-offs between speed and accuracy
- **Image upload**: Support for JPG, JPEG, PNG, BMP, and WEBP formats
- **Live camera capture**: Take photos directly from your webcam for inference
- **Real-time visualization**: Bounding boxes and confidence scores displayed on annotated images
- **Responsive UI**: Industrial dark-themed interface built with Streamlit
- **Model registry**: Extensible architecture — adding new models requires minimal code changes

## Available Models

| Model | Type | Speed | Classes | Description |
|---|---|---|---|---|
| **YOLOv11s** | Object Detection | Fastest | 11 | Single-stage detector optimized for real-time inference |
| **ResNet18** | Multi-Label Classification | Scene-level | 11 | Classifies overall PPE compliance across safety categories |
| **Faster R-CNN** | Object Detection | Most Accurate | 10 | Two-stage detector with highest localization accuracy |

## YOLOv11s and ResNet18 Class

- Helmet / No Helmet
- Gloves / No Gloves
- Vest
- Boots / No Boots
- Goggles / No Goggles
- Person
- None (background)

## R-CNN Class

- respirator
- safety_boots
- safety_glasses
- safety_gloves
- safety_helmet
- welding_helmet
## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Py_11_Multi-Model

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

Place the pre-trained model weights in the `weights/` directory:

```
weights/
├── yolo_weights.pt
├── resnet_weights.pth
└── faster_rcnn_weights.pth
```

Model weights are not included in the repository. Obtain them from the training outputs or contact the project maintainers.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

### How to Use

1. **Select a model** from the sidebar dropdown
2. Wait for the model to load (cached after first use)
3. **Upload an image** via the "UPLOAD IMAGE" tab or **take a photo** using the "LIVE CAMERA" tab
4. View the annotated results with bounding boxes and confidence scores
5. Download the annotated image if needed

## Project Structure

```
Py_11_Multi-Model/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── models/
│   ├── __init__.py             # Model registry (MODEL_REGISTRY)
│   ├── base_model.py           # Abstract base class for all models
│   ├── yolo_model.py           # YOLOv11s wrapper
│   ├── faster_rcnn_model.py    # Faster R-CNN wrapper
│   ├── resnet_model.py         # ResNet18 classifier wrapper
│   └── utils.py                # Shared drawing utilities
├── configs/
│   ├── yolo_config.json        # YOLO model configuration
│   ├── faster_rcnn_config.json # Faster R-CNN configuration
│   └── resnet_config.json      # ResNet configuration
├── weights/                    # Model weights (not in repo)
│   ├── yolo_weights.pt
│   ├── resnet_weights.pth
│   └── faster_rcnn_weights.pth
└── README.md
```

## Adding a New Model

The codebase is designed for easy extension. To add a fourth model:

1. **Create a new model class** in `models/your_model.py` that inherits from `BaseModel` and implements:
   - `name` (property): Display name
   - `description` (property): One-sentence description
   - `task_type` (property): Either `"detection"` or `"classification"`
   - `load_model()`: Load weights and initialize the model
   - `predict(image)` → dict: Return `annotated_image`, `detections`, and `summary`

2. **Create a config file** in `configs/your_model_config.json` with any needed parameters

3. **Register the model** in `models/__init__.py`:
   ```python
   from .your_model import YourModel
   
   MODEL_REGISTRY["YourModel"] = {
       "class": YourModel,
       "config": "configs/your_model_config.json",
       "weights": "weights/your_model_weights.pth",
       "icon": "🚀",
       "badge": "New",
       "description": "Brief description of your model.",
       "task": "Object Detection",  # or "Classification"
       "num_classes": 10,
   }
   ```

The model will automatically appear in the Streamlit UI.

## Requirements

```
streamlit>=1.35.0
torch>=2.1.0
torchvision>=0.16.0
ultralytics>=8.2.0
Pillow>=10.0.0
numpy>=1.26.0
pandas>=2.1.0
```

## Development Notes

- Model loading is cached using `@st.cache_resource` — models load once per session
- Inference transforms for ResNet must match training (no augmentation, optional normalization)
- Detection models draw bounding boxes using color-coded labels
- All models return a standardized result format compatible with the rendering engine

## License

[Add license information here]

## Acknowledgments

- Ultralytics YOLO
- PyTorch / TorchVision
- Streamlit team
