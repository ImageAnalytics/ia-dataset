import unittest
from datetime import datetime
from pathlib import Path

from shapely import Polygon

from ia_dataset.annotation import MLAnnotation, MLAnnotationType, MLAnnotationLabel
from ia_dataset.dataset import MLDataset
from ia_dataset.image import MLImage, MLImageStatus
from ia_dataset.label import MLLabel


class TestMLDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = MLDataset()

    def test_format_time(self):
        timestamp = datetime(2023, 9, 28, 2, 42, 56)
        formatted_time = self.dataset._format_time(timestamp)
        self.assertEqual(formatted_time, '230928T024256000Z')

    def test_label_names(self):
        self.dataset.labels = [MLLabel(None, 'label1'), MLLabel(None, 'label2')]
        self.assertEqual(self.dataset.label_names, ['label1', 'label2'])

    def test_label_dict(self):
        label1 = MLLabel(None, 'label1')
        label2 = MLLabel(None, 'label2')
        self.dataset.labels = [label1, label2]
        self.assertEqual(self.dataset.label_dict(), {'label1': label1, 'label2': label2})

    def test_has_label_name(self):
        self.dataset.labels = [MLLabel(None, 'label1'), MLLabel(None, 'label2')]
        self.assertTrue(self.dataset.has_label_name('label1'))
        self.assertFalse(self.dataset.has_label_name('label3'))

    def test_has_label_names(self):
        self.dataset.labels = [MLLabel(None, 'label1'), MLLabel(None, 'label2')]
        self.assertTrue(self.dataset.has_label_names(['label1', 'label2']))
        self.assertFalse(self.dataset.has_label_names(['label1', 'label3']))

    def test_index_of_label(self):
        self.dataset.labels = [MLLabel(None, 'label1'), MLLabel(None, 'label2')]
        self.assertEqual(self.dataset.index_of_label('label1'), 0)
        self.assertEqual(self.dataset.index_of_label('label2'), 1)

    def test_add_label_if_missing(self):
        label = MLLabel(None, 'label1')
        self.dataset.add_label_if_missing(label)
        self.assertEqual(self.dataset.labels, [label])

    def test_add_labels_if_missing(self):
        labels = [MLLabel(None, 'label1'), MLLabel(None, 'label2')]
        self.dataset.add_labels_if_missing(labels)
        self.assertEqual(self.dataset.labels, labels)

    def test_add_label_if_missing_by_name(self):
        self.dataset.add_label_if_missing_by_name('label1')
        self.assertEqual(self.dataset.label_names, ['label1'])

    def test_add_labels_if_missing_by_name(self):
        self.dataset.add_labels_if_missing_by_name(['label1', 'label2'])
        self.assertEqual(self.dataset.label_names, ['label1', 'label2'])

    def test_update_label_name(self):
        label = MLLabel(None, 'original')
        image = MLImage(0, 0, 0, "0")
        image.annotations = [
            MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('original', 0.5)])]
        self.dataset.labels = [label]
        self.dataset.images = [image]
        self.dataset.update_label_name('original', 'new')
        self.assertEqual(self.dataset.label_names, ['new'])
        self.assertEqual(image.annotations[0].scored_labels[0], MLAnnotationLabel('new', 0.5))

    def test_add_suffix_to_labels(self):
        self.dataset.labels = [MLLabel(None, 'label1'), MLLabel(None, 'label2')]
        image = MLImage(0, 0, 0, "0")
        image.annotations = [
            MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1', 0.5)]),
            MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label2', 0.7)])
        ]
        self.dataset.add_suffix_to_labels('_suffix')
        self.assertEqual(self.dataset.label_names, ['label1_suffix', 'label2_suffix'])
        self.assertEqual(image.annotations[0].scored_labels[0], MLAnnotationLabel('label1_suffix', 0.5))
        self.assertEqual(image.annotations[1].scored_labels[0], MLAnnotationLabel('label2_suffix', 0.7))

    def test_remove_unused_labels(self):
        self.dataset.labels = [MLLabel(None, 'label1'), MLLabel(None, 'label2')]
        image = MLImage(0, 0, 0, "0")
        image.annotations = [
            MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1')])]
        self.dataset.images = [image]
        self.dataset.remove_unused_labels()
        self.assertEqual(self.dataset.label_names, ['label1'])

    def test_update_labels_from_annotations(self):
        image1 = MLImage(0, 0, 0, "0")
        image1.annotations = [
            MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1')])]
        image2 = MLImage(0, 0, 1, "1")
        image2.annotations = [
            MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label2')])]
        self.dataset.images = [image1, image2]
        self.dataset.update_labels_from_annotations()
        self.assertEqual(set(self.dataset.label_names), {'label1', 'label2'})

    def test_add_image(self):
        image = MLImage(0, 0, 0, "0")
        self.dataset.add_image(image)
        self.assertEqual(self.dataset.images, [image])

    def test_remove_unlabelled_images(self):
        image1 = MLImage(0, 0, 0, "0")
        image1.annotations = [
            MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1')])]
        image2 = MLImage(0, 0, 1, "1")
        self.dataset.images = [image1, image2]
        self.dataset.remove_unlabelled_images()
        self.assertEqual(self.dataset.images, [image1])

    def test_remove_labelled_images(self):
        image1 = MLImage(0, 0, 0, "0")
        image1.annotations = [
            MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1')])]
        image2 = MLImage(0, 0, 1, "1")
        self.dataset.images = [image1, image2]
        self.dataset.remove_labelled_images()
        self.assertEqual(self.dataset.images, [image2])

    def test_to_absolute_path(self):
        image1 = MLImage(0, 0, 0, "0")
        self.dataset.images = [image1]
        self.dataset.to_absolute_path('/path/to/dataset')
        self.assertEqual(self.dataset.images[0].path, Path('/path/to/dataset/0'))

    def test_to_relative_path(self):
        image1 = MLImage(0, 0, 0, "/path/to/dataset/0")
        self.dataset.images = [image1]
        self.dataset.to_relative_path('/path/to/dataset')
        self.assertEqual(self.dataset.images[0].path, "0")

    def test_image_dict(self):
        image1 = MLImage(0, 1, 2, "0")
        self.dataset.images = [image1]
        self.assertEqual(self.dataset.image_dict(), {(0, 1, 2): image1})

    def test_filter_by_status(self):
        image1 = MLImage(0, 0, 0, "0", status=MLImageStatus.ANNOTATED)
        image2 = MLImage(1, 0, 0, "1", status=MLImageStatus.IN_PROGRESS)
        self.dataset.images = [image1, image2]
        self.dataset.filter_by_status(MLImageStatus.ANNOTATED)
        self.assertEqual(self.dataset.images, [image1])

    def test_sort_images_by_filename(self):
        image1 = MLImage(0, 0, 0, "1")
        image2 = MLImage(1, 0, 0, "0")
        self.dataset.images = [image1, image2]
        self.dataset.sort_images_by_filename()
        self.assertEqual(self.dataset.images, [image2, image1])

    def test_keep_annotations_with_labels(self):
        image = MLImage(0, 0, 0, "0")
        ann1 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1')])
        ann2 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label2')])
        image.annotations = [ann1, ann2]
        self.dataset.images = [image]
        self.dataset.keep_annotations_with_labels(['label1'])
        self.assertEqual(self.dataset.images[0].annotations, [ann1])

    def test_remove_annotations_with_labels(self):
        image = MLImage(0, 0, 0, "0")
        ann1 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1')])
        ann2 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label2')])
        image.annotations = [ann1, ann2]
        self.dataset.images = [image]
        self.dataset.remove_annotations_with_labels(['label1'])
        self.assertEqual(self.dataset.images[0].annotations, [ann2])

    def test_remove_annotations_below_threshold(self):
        image = MLImage(0, 0, 0, "0")
        ann1 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1', score=0.9)])
        ann2 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label2', score=0.8)])
        image.annotations = [ann1, ann2]
        self.dataset.images = [image]
        self.dataset.remove_annotations_below_threshold(0.85)
        self.assertEqual(self.dataset.images[0].annotations[0].label_names[0].name, 'label1')

    def test_tracks(self):
        image1 = MLImage(0, 0, 0, "0")
        ann1 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1')],
                            is_track=True, track_id=1)
        image1.annotations = [ann1]
        image2 = MLImage(0, 0, 1, "1")
        ann2 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label2')],
                            is_track=True, track_id=1)
        image2.annotations = [ann2]
        self.dataset.images = [image1, image2]
        tracks = self.dataset.tracks()
        self.assertEqual(tracks[(0, 0, 1)], [(ann1, image1), (ann2, image2)])

    def test_split_no_track_large_dataset(self):
        images = [MLImage(i, 0, 0, str(i)) for i in range(20)]
        for i in range(10):  # Add annotations to half of the images
            ann = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label')])
            images[i].annotations = [ann]
        self.dataset.images = images
        train_dataset, val_dataset = self.dataset.split(0.5, False, 0, 0)
        self.assertEqual(len(train_dataset.images), 5)
        self.assertEqual(len(val_dataset.images), 5)
        train_dataset, val_dataset = self.dataset.split(0.5, False, 0, 1)
        self.assertEqual(len(train_dataset.images), 10)
        self.assertEqual(len(val_dataset.images), 10)

    def test_split_with_track_large_dataset(self):
        track_assignment = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        is_track = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
        images = [MLImage(i, 0, 0, str(i)) for i in range(20)]
        for i in range(20):  # Add track annotations
            ann = MLAnnotation(MLAnnotationType.RECT, Polygon(),
                               scored_labels=[MLAnnotationLabel('label')],
                               is_track=is_track[i] == 1,
                               track_id=track_assignment[i])
            images[i].annotations = [ann]
        self.dataset.images = images
        train_dataset, val_dataset = self.dataset.split(0.5, True, 0, 0)
        self.assertEqual(len(train_dataset.images), 6)
        self.assertEqual(len(val_dataset.images), 6)
        train_dataset, val_dataset = self.dataset.split(0.5, True, 0, 1)
        self.assertEqual(len(train_dataset.images), 10)
        self.assertEqual(len(val_dataset.images), 10)

    def test_split_with_track_large_dataset_and_buffer(self):
        track_assignment = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        is_track =         [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
        images = [MLImage(0, 0, i, str(i)) for i in range(20)]
        for i in range(20):  # Add track annotations
            ann = MLAnnotation(MLAnnotationType.RECT,
                               Polygon(),
                               scored_labels=[MLAnnotationLabel('label')],
                               is_track=is_track[i] == 1,
                               track_id=track_assignment[i])
            images[i].annotations = [ann]
        self.dataset.images = images
        train_dataset, val_dataset = self.dataset.split(0.5, True, 1, 0)
        self.assertEqual(len(train_dataset.images), 7)
        self.assertEqual(len(val_dataset.images), 7)
        train_dataset, val_dataset = self.dataset.split(0.5, True, 1, 1)
        self.assertEqual(len(train_dataset.images), 8)
        self.assertEqual(len(val_dataset.images), 8)
        should_be_missing = {4,8,11,16}
        valid = {image.image_id for image in train_dataset.images + val_dataset.images}
        self.assertEqual(should_be_missing.intersection(valid), set())

    def test_label_counts(self):
        image = MLImage(0, 0, 0, "0")
        ann1 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label1')])
        ann2 = MLAnnotation(MLAnnotationType.RECT, Polygon(), scored_labels=[MLAnnotationLabel('label2')])
        image.annotations = [ann1, ann2]
        self.dataset.images = [image]
        self.assertEqual(self.dataset.label_counts(), {"label1": 1, "label2": 1})



if __name__ == '__main__':
    unittest.main()
