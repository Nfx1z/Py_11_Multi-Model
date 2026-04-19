from __future__ import annotations

from PIL import Image

from .base_model import BaseModel


class ResNetModel(BaseModel):
    """ResNet-18 multi-label classification wrapper."""

    # ------------------------------------------------------------------ #
    #  Identity                                                             #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "ResNet18"

    @property
    def description(self) -> str:
        return "Multi-label classifier — scene-level PPE compliance across 10 safety categories."

    @property
    def task_type(self) -> str:
        return "classification"

    # ------------------------------------------------------------------ #
    #  Load                                                                 #
    # ------------------------------------------------------------------ #

    def load_model(self) -> "ResNetModel":
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as T

        self.threshold: float = float(self.config.get("threshold", 0.5))

        # ── 1. Load checkpoint first to read true output size ─────────────
        state = torch.load(self.weights_path, map_location="cpu")
        actual_classes: int = state["fc.weight"].shape[0]

        config_classes: int = self.config["num_classes"]
        if actual_classes != config_classes:
            import warnings
            warnings.warn(
                f"ResNet config says num_classes={config_classes} but "
                f"checkpoint fc.weight has shape [{actual_classes}, 512]. "
                f"Using checkpoint value ({actual_classes})."
            )

        # ── 2. Build model architecture to match checkpoint ───────────────
        self.model = tv_models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, actual_classes)
        self.model.load_state_dict(state)
        self.model.eval()

        # ── 3. Align class names to actual output size ────────────────────
        config_cls: list[str] = self.config["classes"]
        if actual_classes <= len(config_cls):
            self.classes = config_cls[:actual_classes]
        else:
            self.classes = config_cls + [
                f"class_{i}" for i in range(len(config_cls), actual_classes)
            ]

        # ── 4. Build inference transform ──────────────────────────────────
        #
        # Must match the BASE transform used during training, EXCLUDING
        # any augmentations (RandomHorizontalFlip, RandomCrop, ColorJitter,
        # etc.) — those are training-only and must NOT be applied at inference.
        #
        # Training transform was:
        #   Resize((224, 224))
        #   RandomHorizontalFlip()   ← augmentation only, skip at inference
        #   ToTensor()               ← [0,255] PIL → [0,1] float32 CHW
        #   (no Normalize)           ← confirmed: model trained on raw [0,1]
        #
        # Inference transform (what we apply here):
        #   Resize((224, 224))
        #   ToTensor()
        #
        # If normalization IS ever added to training, mirror it here via
        # resnet_config.json:
        #   "normalize": {"mean": [0.485, 0.456, 0.406],
        #                 "std":  [0.229, 0.224, 0.225]}

        input_size: list[int] = self.config.get("input_size", [3, 224, 224])
        h, w = input_size[1], input_size[2]

        norm_cfg = self.config.get("normalize", None)
        if isinstance(norm_cfg, dict):
            normalize_steps = [T.Normalize(norm_cfg["mean"], norm_cfg["std"])]
        else:
            normalize_steps = []  # absent or false → no normalization

        self.transform = T.Compose([
            T.Resize((h, w), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            *normalize_steps,
        ])

        self._torch = torch
        return self

    # ------------------------------------------------------------------ #
    #  Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, image: Image.Image) -> dict:
        # Guard: RGBA / palette / grayscale → RGB
        # A 4-channel tensor silently corrupts the first conv layer and
        # causes one class to always fire above threshold.
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]

        with self._torch.no_grad():
            self.model.eval()   # defensive: stays eval across Streamlit reruns
            logits = self.model(tensor)             # raw logits [1, num_classes]
            scores = self._torch.sigmoid(logits)[0].tolist()  # [num_classes]

        detections = [
            {
                "label": cls,
                "confidence": float(score),
                "detected": float(score) >= self.threshold,
            }
            for cls, score in zip(self.classes, scores)
        ]

        detected_labels = [d["label"] for d in detections if d["detected"]]
        summary = (
            f"Present: {', '.join(detected_labels)}"
            if detected_labels
            else "No PPE classes detected above threshold"
        )

        return {
            "annotated_image": image,
            "detections": detections,
            "summary": summary,
        }