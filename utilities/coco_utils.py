from object_detection.utils import label_map_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

global counter


def _get_dummy_info_string():
    info = {
        "description": "dummy description",
        "url": "dummy url",
        "version": "11.1",
        "year": 2022,
        "contributor": "Martin Horak",
        "date_created": "today"
    }
    return info


def _get_dummy_licence_string():
    licences = [
        {
            "url": "dummy url",
            "id": 1,
            "name": "dummy name"
        }
    ]

    return licences


def _get_categories_string(labelmap_path):
    labelmap = label_map_util.load_labelmap(labelmap_path)
    categories = []
    for item in labelmap.item:
        category = {"supercategory": "UiElement", "id": item.id, "name": item.name}
        categories.append(category)

    return categories


def _get_image_string(data):
    image_id = data.image_name.split('.')[0]
    image_string = {"file_name": data.image_name,
                    "height": data.image_height,
                    "width": data.image_width,
                    "id": image_id}
    return image_string


def convert_absolute_bbox_annotation(xmin, ymin, xmax, ymax):
    bbox_top_left_x = xmin
    bbox_top_left_y = ymin
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    return bbox_top_left_x, bbox_top_left_y, bbox_width, bbox_height


def convert_normalized_coordinates(relative_bbox, image_width, image_height):
    """
    Convert from normalized (range 0-1) (ymin, xmin, ymax, xmax) format to absolute format (top)
    :param relative_bbox: normalized bbox (ymin, xmin, ymax, xmax)
    :param image_width: width of an image
    :param image_height: height of an image
    :return: absolute coordinates of bbox (x_top_left, y_top_left, width, height)
    """

    ymin, xmin, ymax, xmax = relative_bbox

    (xmin, xmax, ymin, ymax) = (xmin * image_width, xmax * image_width, ymin * image_height, ymax * image_height)

    return convert_absolute_bbox_annotation(xmin, ymin, xmax, ymax)


def _get_annotation_string_list(data):
    global counter
    image_id = data.image_name.split('.')[0]

    ann_string_list = []
    for class_id, bbox in zip(data.groundtruth_bboxes.class_ids,
                              data.groundtruth_bboxes.coordinates.coordinates):
        bbox = convert_normalized_coordinates(relative_bbox=bbox,
                                              image_width=data.image_width,
                                              image_height=data.image_height)

        area = bbox[2] * bbox[3]
        annotation_id = counter
        counter = counter + 1

        annotation = {
            "area": area,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": bbox,
            "category_id": int(class_id),
            "id": annotation_id
        }
        ann_string_list.append(annotation)

    return ann_string_list


def generate_ground_truth_coco_data(parsed_detection_data, label_map_path):
    global counter
    counter = 0

    info_string = _get_dummy_info_string()
    licenses_string = _get_dummy_licence_string()
    categories_string = _get_categories_string(label_map_path)

    images_string_list = []
    annotation_string_list = []
    for image_data in parsed_detection_data:
        images_string_list.append(_get_image_string(image_data))
        annotation_string_list.extend(_get_annotation_string_list(image_data))

    coco_annotation = {
        "info": info_string,
        "licenses": licenses_string,
        "images": images_string_list,
        "annotations": annotation_string_list,
        "categories": categories_string,
    }
    return coco_annotation


def _get_prediction_string(data):
    image_id = data.image_name.split('.')[0]

    prediction_string_list = []
    for class_id, bbox, score in zip(data.detected_bboxes.class_ids,
                                     data.detected_bboxes.coordinates.coordinates,
                                     data.detected_bboxes.scores):
        bbox = convert_normalized_coordinates(relative_bbox=bbox,
                                              image_width=data.image_width,
                                              image_height=data.image_height)

        coco_format_prediction = {'image_id': image_id,
                                  'category_id': int(class_id),
                                  'bbox': bbox,
                                  'score': score}

        prediction_string_list.append(coco_format_prediction)

    return prediction_string_list


def generate_detection_coco_data(parsed_data_list, labelmap_path):
    prediction_list = []
    for image_data in parsed_data_list:
        predictions = _get_prediction_string(image_data)
        prediction_list.extend(predictions)

    return prediction_list


def evaluate_from_file(predictions_json_path, groundtruth_json_path):
    cocoGt = COCO(groundtruth_json_path)
    cocoDt = cocoGt.loadRes(predictions_json_path)

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.report_string
