import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np
from marshmallow import fields
from shapely import Polygon

from .base import BaseModel


class PolygonField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return [list(coord) for coord in value.exterior.coords][:-1]

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None
        return Polygon(value)


class MLAnnotationType(Enum):
    POINT = 1
    LINE = 2
    POLYLINE = 3
    RECT = 4
    GRID_CELL = 5
    POLYGON = 6
    IMAGE = 7


@dataclass
class MLAnnotationLabel(BaseModel):
    name: str
    score: float = 1.0
    source: Optional[str] = None


@dataclass
class MLAnnotation(BaseModel):
    type: MLAnnotationType
    polygon: Polygon = field(metadata=dict(marshmallow_field=PolygonField()))
    scored_labels: List[MLAnnotationLabel] = field(default_factory=list)

    is_track: bool = False
    track_id: Optional[int] = None
    track_idx: Optional[int] = None

    id: Optional[int] = None
    is_interpolated: Optional[bool] = False
    is_outside: Optional[bool] = False

    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    @property
    def label_name(self):
        return self.scored_labels[0].name

    @property
    def label_names(self):
        return [label.name for label in self.scored_labels]

    @property
    def score(self):
        return self.scored_labels[0].score

    @property
    def scores(self):
        return [label.score for label in self.scored_labels]

    def bbox_x1y1x2y2(self, as_int=False):
        bounds = self.polygon.bounds
        if as_int:
            bounds = np.round(bounds).astype(int)
        return bounds

    def bbox_xywh(self, as_int=False):
        x1, y1, x2, y2 = self.bbox_x1y1x2y2(as_int)
        return x1, y1, x2 - x1, y2 - y1

    def polygon_to_list(self):
        return [list(coord) for coord in self.polygon.exterior.coords][:-1]

    def polygon_to_flattened_list(self):
        return np.asarray(self.polygon.exterior.coords[:-1]).flatten().tolist()

    def polygon_to_numpy(self, as_int=False) -> np.array:
        polygon_numpy = np.asarray(self.polygon_to_list())
        if as_int:
            polygon_numpy = np.round(polygon_numpy).astype(int)
        return polygon_numpy

    @property
    def centroid(self):
        if self.type == MLAnnotationType.RECT or self.type == MLAnnotationType.GRID_CELL or self.type == MLAnnotationType.POLYGON:
            return list(self.polygon.centroid.coords)[0]
        else:
            raise NotImplementedError(f"Centroid is not implemented for type: {self.type}")

    def iou(self, ann):
        if self.type == MLAnnotationType.RECT or self.type == MLAnnotationType.GRID_CELL or self.type == MLAnnotationType.POLYGON:
            intersect = self.polygon.intersection(ann.polygon).area
            union = self.polygon.union(ann.polygon).area
            return intersect / union
        else:
            raise NotImplementedError(f"IoU not implemented for annotation of type: {self.type}")

    def max_intersection(self, ann):
        if self.type == MLAnnotationType.RECT or self.type == MLAnnotationType.GRID_CELL or self.type == MLAnnotationType.POLYGON:
            intersect = self.polygon.intersection(ann.polygon).area
            return np.max((intersect / self.polygon.area, intersect / ann.polygon.area))
        else:
            raise NotImplementedError(f"Max intersection not implemented for annotation of type: {self.type}")

    def within_bounds(self, x1, y1, x2, y2):
        bounds = Polygon.from_bounds(x1, y1, x2, y2)
        return self.polygon.within(bounds)

    @staticmethod
    def interpolate(start_ann: "MLAnnotation", stop_ann: "MLAnnotation", count, offset):
        if start_ann.type == MLAnnotationType.RECT:
            ann: MLAnnotation = dataclasses.replace(start_ann)
            bbox1 = np.asarray(start_ann.bbox_x1y1x2y2(), np.float32)
            bbox2 = np.asarray(stop_ann.bbox_x1y1x2y2(), np.float32)
            int_bbox = bbox1 + (bbox2 - bbox1) * offset / count
            ann.polygon = Polygon.from_bounds(*int_bbox)
            ann.is_interpolated = True
            ann.track_idx = offset
            return ann
        else:
            raise NotImplementedError(f"Interpolation only implemented for the {MLAnnotationType.RECT} type")
