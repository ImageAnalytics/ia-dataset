from typing import List

import numpy as np

from ia_dataset.annotation import MLAnnotation
from ia_dataset.dataset import MLDataset
from ia_dataset.image import MLImage


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

    # tet

    #                              new_project,
    #                              add_missing_images=True,
    #                              add_missing_annotations=True,
    #                              overwrite_annotation_dimensions=False,
    #                              overwrite_annotation_label=False,
    #                              iou_threshold=0.5,
    #                              add_all_annotations=False):
    # print("Adding data from {}".format(new_project.filename))
    # image_metadata_to_add = []
    # count = 0
    # overlap_count = 0
    # new_image_count = 0
    # annotation_count = 0
    # new_annotation_count = 0
    #
    # ann_dict = dict()
    # overlap_dict = dict()
    # new_dict = dict()
    #
    # missing_key_count = 0
    #
    # # TODO when combining, need to ensure unique sequence ids
    #
    # # Loop through images in new project
    # for key, new_metadata in new_project.image_dict.items():
    #     # If the image is already in the old project ...
    #     if key in self.image_dict:
    #         metadata = self.image_dict[key]
    #         # Check at each the annotation in the new project image metadata ...
    #         for new_annotation in new_metadata.annotations:
    #             annotation_count += 1
    #             if new_annotation.label in ann_dict:
    #                 ann_dict[new_annotation.label] += 1
    #             else:
    #                 ann_dict[new_annotation.label] = 1
    #             overlap = False
    #             # ... against each in the current project,
    #             for annotation in metadata.annotations:
    #                 # and if there is any overlap then flag it
    #                 # print(annotation.iou(new_annotation))
    #                 if annotation.iou(new_annotation) > iou_threshold:
    #                     overlap = True
    #                     overlap_count += 1
    #                     if overwrite_annotation_dimensions:
    #                         annotation.x = new_annotation.x
    #                         annotation.y = new_annotation.y
    #                         annotation.width = new_annotation.width
    #                         annotation.height = new_annotation.height
    #                     if overwrite_annotation_label:
    #                         annotation.label = new_annotation.label
    #                         annotation.score = new_annotation.score
    #                         annotation.annotator = new_annotation.annotator
    #                     if new_annotation.label in overlap_dict:
    #                         overlap_dict[new_annotation.label] += 1
    #                     else:
    #                         overlap_dict[new_annotation.label] = 1
    #                     # break
    #             # but if no overlap, then add the annotation from the new_project
    #             if not add_all_annotations:
    #                 # print(new_annotation.label)
    #                 if overlap is False and add_missing_annotations is True:
    #                     metadata.add_annotation(new_annotation)
    #                     new_annotation_count += 1
    #                     if new_annotation.label in new_dict:
    #                         new_dict[new_annotation.label] += 1
    #                     else:
    #                         new_dict[new_annotation.label] = 1
    #             else:
    #                 metadata.add_annotation(new_annotation)
    #                 new_annotation_count += 1
    #                 if new_annotation.label in new_dict:
    #                     new_dict[new_annotation.label] += 1
    #                 else:
    #                     new_dict[new_annotation.label] = 1
    #     # Image not found - add annotation
    #     elif add_missing_images:
    #         new_image_count += 1
    #         image_metadata_to_add.append(new_metadata)
    #         annotation_count += len(new_metadata.annotations)
    #         new_annotation_count += len(new_metadata.annotations)
    #     else:
    #         missing_key_count += 1
    # print("Results")
    # print(" - annotations processed: {}, {}".format(annotation_count, ann_dict))
    # print(" --- overlapping: {}, {}".format(overlap_count, overlap_dict))
    # print(" --- added: {}, {}".format(new_annotation_count - overlap_count, new_dict))
    # print(" - images added: {}".format(new_image_count))
    # print(" - # keys that could not be matched: {}".format(missing_key_count))
    # for metadata in image_metadata_to_add:
    #     self.add_image(metadata)