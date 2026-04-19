from __future__ import annotations

from PIL import Image, ImageDraw

from .base_model import BaseModel
from .utils import CLASS_COLORS, draw_label


class FasterRCNNModel(BaseModel):
    """Faster R-CNN (ResNet-50 FPN) object detection wrapper."""

    # ------------------------------------------------------------------ #
    #  Identity                                                             #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "Faster R-CNN"

    @property
    def description(self) -> str:
        return "Two-stage detector — highest accuracy, 9 K3-Safety equipment classes."

    @property
    def task_type(self) -> str:
        return "detection"

    # ------------------------------------------------------------------ #
    #  Load                                                                 #
    # ------------------------------------------------------------------ #

    def load_model(self) -> "FasterRCNNModel":
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.id2cat: dict[str, str] = self.config["id2cat"]
        self.score_thresh: float = float(self.config.get("box_score_thresh", 0.5))

        # Load checkpoint first to read the true num_classes from the predictor head
        state = torch.load(self.weights_path, map_location="cpu")
        # cls_score weight shape → (num_classes, in_features)
        actual_classes: int = state["roi_heads.box_predictor.cls_score.weight"].shape[0]

        # weights=None + weights_backbone=None prevents any internet downloads,
        # which is the main cause of slow cold-start on this model
        self.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, actual_classes)

        self.model.load_state_dict(state)
        self.model.eval()
        self._torch = torch
        return self

    # ------------------------------------------------------------------ #
    #  Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, image: Image.Image) -> dict:
        import torchvision.transforms.functional as TF

        tensor = TF.to_tensor(image).unsqueeze(0)
        with self._torch.no_grad():
            output = self.model(tensor)[0]

        detections: list[dict] = []
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for box, label_id, score in zip(
            output["boxes"], output["labels"], output["scores"]
        ):
            if float(score) < self.score_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            label = self.id2cat.get(str(label_id.item()), str(label_id.item()))
            conf = float(score)
            color = CLASS_COLORS[label_id.item() % len(CLASS_COLORS)]

            detections.append(
                {"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]}
            )
            draw_label(draw, x1, y1, x2, y2, label, conf, color)

        return {
            "annotated_image": annotated,
            "detections": detections,
            "summary": f"Detected {len(detections)} object(s)",
        }