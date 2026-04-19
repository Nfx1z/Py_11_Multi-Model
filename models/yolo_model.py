from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .base_model import BaseModel
from .utils import CLASS_COLORS, draw_label


class YOLOModel(BaseModel):
    """YOLOv11s real-time object detection wrapper."""

    # ------------------------------------------------------------------ #
    #  Identity                                                             #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "YOLOv11s"

    @property
    def description(self) -> str:
        return "Real-time single-stage detector — fastest inference, 11 PPE classes."

    @property
    def task_type(self) -> str:
        return "detection"

    # ------------------------------------------------------------------ #
    #  Load                                                                 #
    # ------------------------------------------------------------------ #

    def load_model(self) -> "YOLOModel":
        from ultralytics import YOLO  # lazy import so other models don't need it
        self.model = YOLO(self.weights_path)
        self.class_names: list[str] = self.config["dataset"]["class_names"]
        return self

    # ------------------------------------------------------------------ #
    #  Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, image: Image.Image) -> dict:
        results = self.model(image, verbose=False)[0]

        detections: list[dict] = []
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = (
                self.class_names[cls_id]
                if cls_id < len(self.class_names)
                else str(cls_id)
            )
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

            detections.append(
                {"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]}
            )
            draw_label(draw, x1, y1, x2, y2, label, conf, color)

        return {
            "annotated_image": annotated,
            "detections": detections,
            "summary": f"Detected {len(detections)} object(s)",
        }