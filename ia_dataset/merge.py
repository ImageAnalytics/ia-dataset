from typing import List

import numpy as np

from .annotation import MLAnnotation
from .dataset import MLDataset
from .image import MLImage


def iou_matrix(annotations: List[MLAnnotation],
               annotations_in: List[MLAnnotation]) -> np.array:
    ious = np.zeros((len(annotations), len(annotations_in)))
    for i, ann in enumerate(annotations):
        for j, ann_in in enumerate(annotations_in):
            ious[i, j] = ann.iou(ann_in)
    return ious


def merge_image(image: MLImage,
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


def merge(dataset: MLDataset,
          dataset_merge: MLDataset,
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
                merge_image(image, image_dict_merge[key], iou_threshold)
    else:
        for image, image_in in zip(dataset.images, dataset_merge.images):
            merge_image(image, image_in, iou_threshold)

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
