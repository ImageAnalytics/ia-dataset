import copy
import random
from enum import Enum

import numpy as np
from scipy.ndimage import binary_dilation

from .dataset import MLDataset
from .utils import find_continuous_sequences


class MLSplitMode(Enum):
    IMAGE = 0
    OBJECT = 1
    TRACKED_OBJECT = 2


# TODO test split
def split(dataset: MLDataset,
          val_split: float,
          mode: MLSplitMode,
          track_buffer: int,
          negative_sample_rate: float,
          random_seed=42):
    # Set the random seed
    random.seed(random_seed)

    # If no tracks
    if mode == MLSplitMode.OBJECT:
        # Indices of images with annotations
        has_annotation_idxs = [i for i, image in enumerate(dataset.images) if len(image.annotations) > 0]
        random.shuffle(has_annotation_idxs)
        num_train = int(len(has_annotation_idxs) * (1 - val_split))
        train_idxs = has_annotation_idxs[:num_train]
        val_idxs = has_annotation_idxs[num_train:]

        # If we want some negative examples as well
        if negative_sample_rate > 0.0:
            has_no_annotation_idxs = [i for i, image in enumerate(dataset.images) if len(image.annotations) == 0]
            random.shuffle(has_no_annotation_idxs)
            num_negative = min(int(negative_sample_rate * len(has_annotation_idxs)), len(has_no_annotation_idxs))
            num_train = int(num_negative * (1 - val_split))
            train_idxs.extend(has_no_annotation_idxs[:num_train])
            val_idxs.extend(has_no_annotation_idxs[num_train:num_negative])

    elif mode == MLSplitMode.TRACKED_OBJECT:
        # Array of 0 for image without a track, 1 with a track
        track_vec = np.zeros(len(dataset.images))
        for i, image in enumerate(dataset.images):
            for ann in image.annotations:
                if ann.is_track:
                    track_vec[i] = 1

        # Dilate by track buffer
        strel = np.ones(track_buffer * 2 + 1)
        dilated_track_vec = binary_dilation(track_vec, structure=strel).astype(int)

        # Add original, now 1 = buffer, 2 = track
        track_vec += dilated_track_vec

        # Tracks
        track_idx_list = find_continuous_sequences(track_vec, 2)
        print(track_idx_list)
        random.shuffle(track_idx_list)
        num_train = int(len(track_idx_list) * (1 - val_split))

        # Train indices - split by sets instead of individual images
        train_idxs = []
        for idxs in track_idx_list[:num_train]:
            train_idxs.extend(idxs)

        # Val indices
        val_idxs = []
        for idxs in track_idx_list[num_train:]:
            val_idxs.extend(idxs)

        # Negatives
        if negative_sample_rate > 0.0:
            # Find all the images without a track and outside the buffer zone
            neg_idx_list = find_continuous_sequences(track_vec, 0)
            neg_idxs = []
            for idxs in neg_idx_list:
                neg_idxs.extend(idxs)

            # Randomise the list
            random.shuffle(neg_idxs)

            # Choose some to add to the training and validation sets
            # The number chosen is the negative sample rate time the number of images with track annotations
            total_pos = len(train_idxs) + len(val_idxs)
            num_negative = min(int(negative_sample_rate * total_pos), len(neg_idxs))
            num_train = int(num_negative * (1 - val_split))
            train_idxs.extend(neg_idxs[:num_train])
            val_idxs.extend(neg_idxs[num_train:num_negative])

    else:
        # Remove all object annotations
        for image in dataset.images:
            image.annotations = []

        # Indices of images with annotations
        has_annotation_idxs = [i for i, image in enumerate(dataset.images) if len(image.image_scored_labels) > 0]
        random.shuffle(has_annotation_idxs)
        num_train = int(len(has_annotation_idxs) * (1 - val_split))
        train_idxs = has_annotation_idxs[:num_train]
        val_idxs = has_annotation_idxs[num_train:]

    # Sort
    train_idxs = sorted(train_idxs)
    val_idxs = sorted(val_idxs)

    # Create new datasets
    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)
    train_dataset.images = [train_dataset.images[i] for i in train_idxs]
    val_dataset.images = [val_dataset.images[i] for i in val_idxs]

    return train_dataset, val_dataset