from dataset_objects.bbox import Bboxes


class Prediction:
    def __init__(self, image_name, image_width, image_height, groundtruth_bboxes: Bboxes, detected_bboxes: Bboxes):
        self.image_name = image_name
        self.image_width = image_width
        self.image_height = image_height
        self.groundtruth_bboxes = groundtruth_bboxes
        self.detected_bboxes = detected_bboxes