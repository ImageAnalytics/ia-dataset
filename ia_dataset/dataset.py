import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .annotation import MLAnnotation
from .base import BaseModel
from .image import MLImage, MLImageStatus
from .label import MLLabel


@dataclass
class MLDataset(BaseModel):
    labels: List[MLLabel] = field(default_factory=list)
    images: List[MLImage] = field(default_factory=list)

    def _format_time(self, timestamp: datetime):
        return timestamp.strftime("%y%m%dT%H%M%S%f")[:-3] + "Z"

    def image_dict(self, use_filename=False) -> Dict[Tuple[Union[int, str], Union[int, str], Union[int, str]], MLImage]:
        return {image.get_key(use_filename): image for image in self.images}

    """
    Labels
    """

    @property
    def label_names(self) -> List[str]:
        return [label.name for label in self.labels]

    def label_dict(self) -> Dict[str, MLLabel]:
        return {label.name: label for label in self.labels}

    def has_label_name(self, label: str) -> bool:
        return label in self.label_names

    def has_label_names(self, labels: list) -> bool:
        for label in labels:
            if not self.has_label_name(label):
                return False
        return True

    def index_of_label(self, label) -> int:
        return self.label_names.index(label)

    def add_label_if_missing(self, label: MLLabel):
        self.add_labels_if_missing([label])

    def add_labels_if_missing(self, labels: List[MLLabel]):
        label_names = self.label_names
        for label in labels:
            if label.name not in label_names:
                self.labels.append(label)

    def add_label_if_missing_by_name(self, label: str):
        self.add_labels_if_missing_by_name([label])

    def add_labels_if_missing_by_name(self, labels: List[str]):
        label_names = self.label_names
        for label in labels:
            if label not in label_names:
                self.labels.append(MLLabel(None, label))

    def update_label_name(self, original, new):
        # Global label list
        self.labels = [label for label in self.labels if label.name != original]
        self.add_label_if_missing_by_name(new)

        # Image labels
        for image in self.images:
            for ann in image.annotations:
                # If new label already exists, just remove the original
                if new in ann.label_names:
                    ann.scored_labels = [label for label in ann.scored_labels if label.name != original]
                # Otherwise, replace the original with the new
                else:
                    for label in ann.scored_labels:
                        if label.name == original:
                            label.name = new
        self.update_labels_from_annotations()

    def add_suffix_to_labels(self, suffix: str):
        for label in self.labels:
            label.name += suffix
        for image in self.images:
            for ann in image.annotations:
                for label in ann.scored_labels:
                    label.name += suffix
        self.update_labels_from_annotations()

    def remove_unused_labels(self):
        label_names = set()
        for image in self.images:
            for label in image.labels:
                label_names.add(label)
        self.labels = [label for label in self.labels if label.name in label_names]

    def update_labels_from_annotations(self):
        labels = set()
        for image in self.images:
            for label in image.labels:
                labels.add(label)
        self.add_labels_if_missing_by_name(list(labels))

    """
    Images
    """

    def add_image(self, image: MLImage):
        self.images.append(image)

    def remove_unlabelled_images(self):
        self.images = [im for im in self.images if len(im.annotations) > 0]

    def remove_labelled_images(self):
        self.images = [im for im in self.images if len(im.annotations) == 0]

    def to_absolute_path(self, path):
        path = Path(path)
        for image in self.images:
            image.path = path / image.path

    def to_relative_path(self, path):
        path = Path(path)
        for image in self.images:
            image.path = str(Path(image.path).relative_to(path))

    def add_containing_folder_to_filename(self, folder):
        for image in self.images:
            image.path = os.path.join(folder, image.path)

    def filter_by_status(self, min_status: MLImageStatus = MLImageStatus.ANNOTATED):
        valid_status = [
            MLImageStatus.NOT_STARTED,
            MLImageStatus.IN_PROGRESS,
            MLImageStatus.ANNOTATED,
            MLImageStatus.VALIDATED]
        # if min_status not in valid_status:
        #     raise ValueError(f"min_status ({min_status}) must be one of {valid_status}")
        min_status_level = valid_status.index(min_status)
        images = []
        for image in self.images:
            if image.status not in valid_status:
                raise ValueError(f"image status ({image.status}) must be one of {valid_status}, {image.path}")
            status_level = valid_status.index(image.status)
            if status_level >= min_status_level:
                images.append(image)
        self.images = images

    def sort_images_by_filename(self):
        self.images.sort(key=lambda x: x.path)
        for idx, image in enumerate(self.images):
            image.image_id = idx

    """
    Annotations
    """

    def keep_annotations_with_labels(self, labels: List[str]):
        for image in self.images:
            anns = []
            for ann in image.annotations:
                new_labels = []
                for lab in ann.scored_labels:
                    if lab.name in labels:
                        new_labels.append(lab)
                ann.label_names = new_labels
                if len(ann.label_names) > 0:
                    anns.append(ann)
            image.annotations = anns

    def remove_annotations_with_labels(self, labels: List[str]):
        for image in self.images:
            anns = []
            for ann in image.annotations:
                new_labels = []
                for lab in ann.scored_labels:
                    if lab.name not in labels:
                        new_labels.append(lab)
                ann.label_names = new_labels
                if len(ann.label_names) > 0:
                    anns.append(ann)
            image.annotations = anns

    def remove_annotations_below_threshold(self, threshold: float):
        for image in self.images:
            anns = []
            for ann in image.annotations:
                new_labels = []
                for lab in ann.scored_labels:
                    if lab.score >= threshold:
                        new_labels.append(lab)
                ann.label_names = new_labels
                if len(ann.label_names) > 0:
                    anns.append(ann)
            image.annotations = anns

    """
    Tracks
    """

    def tracks(self) -> Dict[Tuple[Union[int,str], Union[int,str], int], List[Tuple[MLAnnotation, MLImage]]]:
        tracks = dict()
        # Group into tracks
        for image in self.images:
            for ann in image.annotations:
                if ann.is_track:
                    # Project and task are included in the track id
                    track_id = (image.project_id, image.task_id, ann.track_id)
                    if track_id not in tracks:
                        tracks[track_id] = []
                    tracks[track_id].append((ann, image))
        # Make sure ordered by frame number
        tracks = {key: sorted(value, key=lambda x: x[1].image_id) for key, value in tracks.items()}
        return tracks


    """
    Sanitising
    """

    def reindex_images(self):
        # Frame numbers
        for idx, image in enumerate(self.images):
            image.project_id = 0
            image.task_id = 0
            image.image_id = idx

    def reindex_tracks(self):
        for idx, track_data in enumerate(self.tracks().values()):
            for (ann, image) in track_data:
                ann.track_id = idx

    """
    Summaries
    """

    def label_counts(self):
        self.update_labels_from_annotations()
        counts = {k.name: 0 for k in self.labels}
        for image in self.images:
            for ann in image.annotations:
                for label in ann.label_names:
                    counts[label] += 1
        return counts

    def image_label_counts(self):
        self.update_labels_from_annotations()
        counts = {k.name: 0 for k in self.labels}
        for image in self.images:
            for ann in image.image_scored_labels:
                counts[ann.name] += 1
        return counts

    def annotation_counts(self):
        counts = {"0": 0,
                  "1-10": 0,
                  "11-100": 0,
                  "100+": 0}
        for image in self.images:
            anns = len(image.annotations)
            if anns == 0:
                counts["0"] += 1
            elif anns <= 10:
                counts["1-10"] += 1
            elif anns <= 100:
                counts["11-100"] += 1
            else:
                counts["100+"] += 1
        return counts

    def summary(self):
        print("-" * 80)
        print("Project summary")
        print("Images:")
        print(f"- count: {len(self.images)}")
        print("Labels:")
        for label in self.labels:
            print(f"- {label.name} - id: {label.id}, colour: {label.colour}")
        print("Object annotation count:")
        counts = self.label_counts()
        for label in self.labels:
            print(f"- {label.name}: {counts[label.name]}")
        print(f"- TOTAL: {sum(counts.values())}")
        print("Image label count:")
        counts = self.image_label_counts()
        for label in self.labels:
            print(f"- {label.name}: {counts[label.name]}")
        print(f"- TOTAL: {sum(counts.values())}")
        print("Object annotations per image:")
        for rng, count in self.annotation_counts().items():
            print(f"- {rng}: {count}")
        print("-" * 80)
        print()

    """
    Load
    """

    @staticmethod
    def from_directory(path, extension="*.jpg") -> "MLDataset":
        filenames = Path(path).rglob(extension)
        dataset = MLDataset()
        for i, filename in enumerate(filenames):
            dataset.images.append(MLImage(0, 0, i, str(filename)))
        return dataset


