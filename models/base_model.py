from abc import ABC, abstractmethod
from PIL import Image
import json


class BaseModel(ABC):
    """
    Abstract base class for all detection/classification models.

    To add a new model:
      1. Create a new file, e.g. models/my_new_model.py
      2. Subclass BaseModel and implement all abstract methods/properties
      3. Register it in models/__init__.py → MODEL_REGISTRY
    That's it — the app picks it up automatically.
    """

    def __init__(self, config_path: str, weights_path: str):
        self.config_path = config_path
        self.weights_path = weights_path
        self.config = self._load_config()
        self.model = None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    #  Abstract interface — every model MUST implement these              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_model(self) -> "BaseModel":
        """Load weights and prepare the model.  Must return `self`."""
        ...

    @abstractmethod
    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL Image.

        Returns a dict with the following keys:
          - annotated_image (PIL.Image): image with boxes / labels drawn
          - detections (list[dict]):
              For detection models each item has:
                  label (str), confidence (float), bbox (list[int] x1y1x2y2)
              For classification models each item has:
                  label (str), confidence (float), detected (bool)
          - summary (str): one-line human-readable result
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name, e.g. 'YOLOv11s'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description shown in the sidebar."""
        ...

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Either 'detection' or 'classification'."""
        ...

    # ------------------------------------------------------------------ #
    #  Convenience                                                          #
    # ------------------------------------------------------------------ #

    def is_loaded(self) -> bool:
        return self.model is not None