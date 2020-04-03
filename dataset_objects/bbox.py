import enum


class CoordinatesType(enum.Enum):
    x_y_width_height = 1
    ymin_xmin_ymax_xmax = 2


class BboxCoordinates:

    def __init__(self, coordinates_list, is_relative: bool, coordinate_type: CoordinatesType):
        self.coordinates = coordinates_list
        self.is_relative = is_relative
        self.type = coordinate_type


class Bboxes:

    def __init__(self, coordinates: BboxCoordinates, scores, class_ids, is_groundtruth):
        self.coordinates = coordinates
        self.scores = scores
        self.class_ids = class_ids
        self.is_groundtruth = is_groundtruth