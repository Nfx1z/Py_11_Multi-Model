"""
Model registry — the ONLY file you need to edit to add a new model.

Steps to add Model #4:
  1. Create  models/my_new_model.py  subclassing BaseModel
  2. Add an entry to MODEL_REGISTRY below
  3. Done — the Streamlit app picks it up automatically.
"""
from __future__ import annotations

from .yolo_model import YOLOModel
from .resnet_model import ResNetModel
from .faster_rcnn_model import FasterRCNNModel
from .base_model import BaseModel

# ── Registry ──────────────────────────────────────────────────────────── #
# Each entry:
#   class      : BaseModel subclass
#   config     : path to the JSON config (relative to project root)
#   weights    : path to the weights file (relative to project root)
#   icon       : emoji shown in the sidebar
#   badge      : short speed/accuracy label
#   description: one sentence shown in the model card
#   task       : human-readable task label
#   num_classes: used for display only
# ─────────────────────────────────────────────────────────────────────── #

MODEL_REGISTRY: dict[str, dict] = {
    "YOLOv11s": {
        "class": YOLOModel,
        "config": "configs/yolo_config.json",
        "weights": "weights/yolo_weights.pt",
        "icon": "⚡",
        "badge": "Fastest",
        "description": "Single-stage detector optimised for real-time inference.",
        "task": "Object Detection",
        "num_classes": 11,
    },
    "ResNet18": {
        "class": ResNetModel,
        "config": "configs/resnet_config.json",
        "weights": "weights/resnet_weights.pth",
        "icon": "🔍",
        "badge": "Classifier",
        "description": "Multi-label scene classification for PPE compliance.",
        "task": "Multi-Label Classification",
        "num_classes": 10,
    },
    "Faster R-CNN": {
        "class": FasterRCNNModel,
        "config": "configs/faster_rcnn_config.json",
        "weights": "weights/faster_rcnn_weights.pth",
        "icon": "🎯",
        "badge": "Most Accurate",
        "description": "Two-stage detector with the highest localisation accuracy.",
        "task": "Object Detection",
        "num_classes": 9,
    },
    # ── Add Model #4 here ──────────────────────────────────────────────
    # "MyNewModel": {
    #     "class": MyNewModel,
    #     "config": "configs/my_new_model_config.json",
    #     "weights": "weights/my_new_model_weights.pth",
    #     "icon": "🚀",
    #     "badge": "New",
    #     "description": "Short description of what it does.",
    #     "task": "Object Detection",
    #     "num_classes": 12,
    # },
}


def get_model(model_name: str) -> BaseModel:
    """Instantiate and return the model (weights NOT loaded yet)."""
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    entry = MODEL_REGISTRY[model_name]
    return entry["class"](entry["config"], entry["weights"])