from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from .annotation import MLAnnotation, MLAnnotationLabel
from .base import BaseModel

class MLImageStatus(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    ANNOTATED = 2
    VALIDATED = 3


@dataclass
class MLImage(BaseModel):
    project_id: Union[int, str]
    task_id: Union[int, str]
    image_id: Union[int, str]
    path: str
    container: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    timestamp: Optional[datetime] = None
    status: Optional[MLImageStatus] = None
    annotations: List[MLAnnotation] = field(default_factory=list)
    image_scored_labels: List[MLAnnotationLabel] = field(default_factory=list)

    def get_key(self, use_filename=False):
        if use_filename:
            return self.project_id, self.task_id, str(Path(self.path).stem)
        else:
            return self.project_id, self.task_id, self.image_id

    """
    Labels
    """
    @property
    def labels(self):
        label_names = set()
        for scored_label in self.image_scored_labels:
            label_names.add(scored_label.name)
        for ann in self.annotations:
            label_names.update(ann.label_names)
        return list(label_names)

    def has_label(self, label: str):
        return label in self.labels

    def has_labels(self, labels: list):
        for label in labels:
            if self.has_label(label):
                return True
        return False

    @property
    def image_labels(self):
        return [label.name for label in self.image_scored_labels]

    def has_image_label(self, label: str):
        return label in self.image_labels

    def has_image_labels(self, labels: list):
        for label in labels:
            if self.has_image_label(label):
                return True
        return False

    """
    Time parsing
    """
    @staticmethod
    def time_from_filename(path, mode=0):
        fn = str(Path(path).stem)
        if mode == 0:
            fstr = '%Y%m%dT%H%M%S%fZ'
            dt = fn[-len(fstr)-4:-4]
            return datetime.strptime(dt, '%Y%m%dT%H%M%S%fZ')
        if mode == 1:
            fstr = '%Y%m%d_%H%M%S%f'
            dt = fn[-12-18:-12]
            return datetime.strptime(dt, fstr)