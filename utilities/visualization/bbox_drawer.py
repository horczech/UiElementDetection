

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import numpy as np


class BboxDrawer:


    def __init__(self, label_map_path):
        self.category_index = self._load_labelmap(label_map_path)
        self.class_dict = label_map_util.get_label_map_dict(label_map_path)

        a=5

    def _load_labelmap(self, path):
        label_map = label_map_util.load_labelmap(path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        return category_index

    def draw_object_from_annotations(self, image, bboxes, class_indexes):

        detection_scores = np.ones(len(class_indexes))

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.asarray(bboxes),
            np.asarray(class_indexes),
            detection_scores,
            self.category_index,
            use_normalized_coordinates=False,
            line_thickness=8)
        return image

    def draw_detections(self, image, bboxes, class_indexes, detection_scores, use_normalized_coordinates=False):

        if detection_scores is None:
            detection_scores = np.ones(len(class_indexes))

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.asarray(bboxes),
            np.asarray(class_indexes),
            detection_scores,
            self.category_index,
            use_normalized_coordinates=use_normalized_coordinates,
            line_thickness=8)
        return image



if __name__ == '__main__':
    bbox_drawer = BboxDrawer(label_map_path=r"C:\Code\TrainedModels\label_map.pbtxt")