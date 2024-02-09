from enum import Enum, auto
from typing import List

import numpy as np

from .annotation import MLAnnotation
from .dataset import MLDataset
from .image import MLImage


class MLMergeMode(Enum):
    REPLACE = auto()
    BOTH = auto()
    MISSING = auto()
    IF_NONE_IN_IMAGE = auto()


def iou_matrix(annotations: List[MLAnnotation],
               annotations_in: List[MLAnnotation]) -> np.array:
    ious = np.zeros((len(annotations), len(annotations_in)))
    for i, ann in enumerate(annotations):
        for j, ann_in in enumerate(annotations_in):
            ious[i, j] = ann.iou(ann_in)
    return ious


def merge_objects_in_image(image: MLImage,
                           image_in: MLImage,
                           iou_threshold: float = 0.5):
    # Get the ious of all the annotations
    ious = iou_matrix(image.annotations, image_in.annotations)

    matched = np.array([False] * len(image_in.annotations))
    while np.any(ious > iou_threshold):
        i, j = np.unravel_index(np.argmax(ious), ious.shape)
        matched[j] = True
        ious[i, :] = 0
        ious[:, j] = 0

    idxs = np.where(matched == False)
    for i in idxs[0]:
        image.annotations.append(image_in.annotations[i])


def merge_image(image: MLImage,
                image_in: MLImage,
                mode: MLMergeMode,
                iou_threshold: float = 0.5):
    if mode == MLMergeMode.REPLACE:
        # Replace all object annotations
        image.annotations = image_in.annotations
        # Replace image labels
        image.labels = image_in.labels
    if mode == MLMergeMode.BOTH:
        # Add all object annotations
        image.annotations.extend(image_in.annotations)
        # Add image labels if not already there
        existing_labels = set([label.name for label in image.labels])
        for label in image_in.labels:
            if label.name not in existing_labels:
                image.labels.append(label)
    elif mode == MLMergeMode.MISSING:
        # Add missing object annotations if their iou with existing annotations is below the threshold
        merge_objects_in_image(image, image_in, iou_threshold)
        # Add image labels if not already there
        existing_labels = set([label.name for label in image.labels])
        for label in image_in.labels:
            if label.name not in existing_labels:
                image.labels.append(label)
    elif mode == MLMergeMode.IF_NONE_IN_IMAGE:
        # Add object annotations if there are none in the image
        if len(image.annotations) == 0:
            image.annotations.extend(image_in.annotations)
        # Add image labels if none exist
        if len(image.labels) == 0:
            image.labels.extend(image_in.labels)


def merge(dataset: MLDataset,
          dataset_merge: MLDataset,
          mode: MLMergeMode,
          iou_threshold: float = 0.5,
          as_nv: bool = True,
          use_key: bool = True):

    # Change the merged in labels to NV if desired
    if as_nv:
        for label in dataset_merge.labels:
            if not label.name.endswith("_NV"):
                dataset_merge.update_label_name(label.name, label.name + "_NV")

    # Merge
    if use_key:
        image_dict = dataset.image_dict()
        image_dict_merge = dataset_merge.image_dict()

        for key, image in image_dict.items():
            if key in image_dict_merge:
                merge_image(image, image_dict_merge[key], mode, iou_threshold)

    else:
        for image, image_in in zip(dataset.images, dataset_merge.images):
            merge_image(image, image_in, mode, iou_threshold)

    dataset.update_labels_from_annotations()


def remove_overlapping_image(image: MLImage, iou_threshold: float = 0.5):
    annotations = sorted(image.annotations, key=lambda x: x.score, reverse=True)

    # Initialize a list to keep track of non-overlapping annotations
    non_overlapping_annotations = []

    while annotations:
        # Take the annotation with the highest score
        current_annotation = annotations.pop(0)
        non_overlapping_annotations.append(current_annotation)

        # Remove annotations that overlap with the current one beyond the IoU threshold
        annotations = [
            annotation for annotation in annotations
            if current_annotation.iou(annotation) <= iou_threshold
        ]

    # Replace the original list of annotations with non-overlapping ones
    image.annotations = non_overlapping_annotations


def remove_overlapping(dataset: MLDataset, iou_threshold: float = 0.5):
    for image in dataset.images:
        remove_overlapping_image(image, iou_threshold)
