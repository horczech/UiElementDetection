import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

from PIL import Image
from os import path

from dataset_objects.bbox import CoordinatesType, BboxCoordinates, Bboxes
from dataset_objects.prediction import Prediction


class DetectionParser:

    def __init__(self, detection_record_path, label_map_path, image_dir):
        self.detection_record_path = detection_record_path
        self.label_map_path = label_map_path
        self.label_map_dict = label_map_util.get_label_map_dict(label_map_path)
        self.image_dir = image_dir

    def parse(self):
        record_iterator = tf.python_io.tf_record_iterator(path=self.detection_record_path)
        data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

        image_idx = 0
        parsed_data_list = []
        for string_record in record_iterator:
            image_idx += 1
            if image_idx % 100 == 0:
                print(f"Processing image No.{image_idx}")

            example = tf.train.Example()
            example.ParseFromString(string_record)
            decoded_dict = data_parser.parse(example)

            if not decoded_dict:
                print(f'Skipped image No.{image_idx}')
                continue

            image_name = decoded_dict[standard_fields.InputDataFields.key]

            groundtruth_boxes = decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
            groundtruth_classes = decoded_dict[standard_fields.InputDataFields.groundtruth_classes]

            detection_scores = decoded_dict[standard_fields.DetectionResultFields.detection_scores]
            detection_classes = decoded_dict[standard_fields.DetectionResultFields.detection_classes]
            detection_boxes = decoded_dict[standard_fields.DetectionResultFields.detection_boxes]

            img_width, img_height = self._get_image_size(image_name)

            gt = Bboxes(coordinates=BboxCoordinates(coordinates_list=groundtruth_boxes,
                                                    is_relative=True,
                                                    coordinate_type=CoordinatesType.ymin_xmin_ymax_xmax),
                        class_ids=groundtruth_classes,
                        scores=None,
                        is_groundtruth=True)

            dt = Bboxes(coordinates=BboxCoordinates(coordinates_list=detection_boxes,
                                                    is_relative=True,
                                                    coordinate_type=CoordinatesType.ymin_xmin_ymax_xmax),
                        class_ids=detection_classes,
                        scores=detection_scores,
                        is_groundtruth=False)

            parsed_data = Prediction(image_name=image_name,
                                     image_width=img_width,
                                     image_height=img_height,
                                     groundtruth_bboxes=gt,
                                     detected_bboxes=dt)
            parsed_data_list.append(parsed_data)

        print(f'Successfully finished parsing')
        return parsed_data_list

    def _get_image_size(self, image_name):
        img_path = path.join(self.image_dir, image_name)

        return Image.open(img_path).size


