from dataclasses import dataclass
from pprint import pprint
from typing import List, Dict

from shapely import Polygon

from .annotation import MLAnnotation, MLAnnotationType, MLAnnotationLabel
from .base import BaseModel
from .dataset import MLDataset
from .image import MLImage


@dataclass
class CocoImage(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    license: int = ""
    date_captured: str = ""


@dataclass
class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: List[float]
    iscrowd: int = 0


@dataclass
class CocoCategory(BaseModel):
    id: int
    name: str
    supercategory: str = "none"


@dataclass
class CocoInfo(BaseModel):
    year: int
    version: str
    description: str = ""
    contributor: str = ""
    url: str = ""
    date_created: str = ""


@dataclass
class CocoLicense(BaseModel):
    id: int
    name: str
    url: str


@dataclass
class CocoDataset(BaseModel):
    info: CocoInfo
    licenses: List[CocoLicense]
    images: List[CocoImage]
    annotations: List[CocoAnnotation]
    categories: List[CocoCategory]

    @staticmethod
    def load(path) -> "CocoDataset":
        return CocoDataset.open(path)

    def save(self, path):
        self.save(path)

    def to_dataset(self) -> MLDataset:
        return CocoConverter.from_coco(self)

    @staticmethod
    def from_dataset(dataset: MLDataset):
        # TODO needs update to use class_schema
        return CocoConverter.to_coco(dataset, [])


class CocoConverter:
    @staticmethod
    def from_coco(coco_data: CocoDataset) -> MLDataset:
        """
        Convert a COCO dataset to an MLDataset
        """
        dataset = MLDataset()

        # If more than one supercategory, use supercategory + name as MLDataset label
        supercategories = set([c.supercategory for c in coco_data.categories])
        # print(supercategories)
        if len(supercategories) > 1:
            label_dict = {c.id: f"{c.supercategory} - {c.name}" for c in coco_data.categories}
        else:
            label_dict = {c.id: c.name for c in coco_data.categories}

        for coco_image in coco_data.images:
            image = MLImage(
                project_id=0,
                task_id=0,
                image_id=coco_image.id,
                path=coco_image.file_name,
                container=None,
                width=coco_image.width,
                height=coco_image.height,
                timestamp=None,
                status=None,
                annotations=[]
            )
            dataset.images.append(image)

        image_dict = dataset.image_dict()
        for coco_annotation in coco_data.annotations:
            annotation = MLAnnotation(
                type=MLAnnotationType.RECT,
                labels=[MLAnnotationLabel(label_dict[coco_annotation.category_id])],
                polygon=Polygon.from_bounds(
                    coco_annotation.bbox[0],
                    coco_annotation.bbox[1],
                    coco_annotation.bbox[0] + coco_annotation.bbox[2],
                    coco_annotation.bbox[1] + coco_annotation.bbox[3]),
                is_track=False,
                track_id=None,
                track_idx=None,
                is_interpolated=False,
                is_outside=False,
                metadata=None
            )
            image_dict[(0, 0, coco_annotation.image_id)].annotations.append(annotation)

        dataset.update_labels_from_annotations()
        dataset.remove_unused_labels()
        return dataset

    @staticmethod
    def to_coco(project: MLDataset, labels):
        """
        Convert an MLDataset to a COCO dataset
        """
        info = {}
        licences = []
        images = []
        annotations = []
        categories = []

        category_id = 1
        for label in labels:
            coco_category = {}
            coco_category["supercategory"] = "none"
            coco_category["id"] = category_id
            coco_category["name"] = label
            categories.append(coco_category)
            category_id += 1

        image_id = 1
        annotation_id = 1
        for metadata in project.images:
            coco_image_data = {}
            coco_image_data["filename"] = metadata.path
            coco_image_data["id"] = image_id

            for ann in metadata.annotations:
                x,y,w,h = ann.bbox_xywh()
                coco_annotation = {}
                coco_annotation["image_id"] = image_id
                coco_annotation["category_id"] = labels.index(ann.label_name) + 1
                coco_annotation["id"] = annotation_id
                coco_annotation["bbox"] = ann.bbox_xywh()
                coco_annotation["score"] = ann.score
                coco_annotation["segmentation"] = []
                coco_annotation["iscrowd"] = 0
                coco_annotation["area"] = w * h
                annotations.append(coco_annotation)
                annotation_id += 1

            images.append(coco_image_data)
            image_id += 1

        coco = {
            "info": info,
            "licences": licences,
            "categories": categories,
            "images": images,
            "annotations": annotations
        }

        return coco


if __name__ == "__main__":
    coco_dataset = CocoDataset.load("../../datasets/Fish.v1-416x416.coco/train/_annotations.coco.json")
    pprint(coco_dataset)
    dataset = coco_dataset.to_dataset()
    pprint(dataset)

    coco_dataset = CocoDataset.load("../../datasets/Boggle Boards.v4-416x416-auto-orient.coco/export/_annotations.coco.json")
    pprint(coco_dataset)
    dataset = coco_dataset.to_dataset()
    pprint(dataset)
