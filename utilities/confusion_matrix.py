import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string('label_map', None, 'Path to the label map')
flags.DEFINE_string('detections_record', None, 'Path to the detections record file')
flags.DEFINE_string('output_path', None, 'Path to the output the results in a csv.')

FLAGS = flags.FLAGS

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5


def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)


def process_detections(parsed_detection_data, categories):
    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))

    image_index = 0
    for image_index, prediction in enumerate(parsed_detection_data):
        groundtruth_boxes = prediction.groundtruth_bboxes.coordinates.coordinates
        groundtruth_classes = prediction.groundtruth_bboxes.class_ids

        detection_scores = prediction.detected_bboxes.scores
        detection_classes = prediction.detected_bboxes.class_ids[detection_scores >= CONFIDENCE_THRESHOLD]
        detection_boxes = prediction.detected_bboxes.coordinates.coordinates[detection_scores >= CONFIDENCE_THRESHOLD]

        matches = []
        if image_index % 100 == 0:
            print("Processed %d images" % (image_index))

        for i in range(len(groundtruth_boxes)):
            for j in range(len(detection_boxes)):
                iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

                if iou > IOU_THRESHOLD:
                    matches.append([i, j, iou])

        matches = np.array(matches)
        if matches.shape[0] > 0:
            # Sort list of matches by descending IOU so we can remove duplicate detections
            # while keeping the highest IOU entry.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate detections from the list.
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            # Sort the list again by descending IOU. Removing duplicates doesn't preserve
            # our previous sort.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate ground truths from the list.
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        for i in range(len(groundtruth_boxes)):
            if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                confusion_matrix[groundtruth_classes[i] - 1][
                    detection_classes[int(matches[matches[:, 0] == i, 1][0])] - 1] += 1
            else:
                confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1

        for i in range(len(detection_boxes)):
            if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
                confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1

    print(f"Processed {image_index} images")
    return confusion_matrix


def get_evaluations(confusion_matrix, categories):
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    results = []

    class_list = []
    for i in range(len(categories)):
        id = categories[i]["id"] - 1
        name = categories[i]["name"]

        class_list.append(name)

        total_target = np.sum(confusion_matrix[id, :])
        total_predicted = np.sum(confusion_matrix[:, id])

        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)

        results.append({'category': name, 'precision_@{}IOU'.format(IOU_THRESHOLD): precision,
                        'recall_@{}IOU'.format(IOU_THRESHOLD): recall})

    evaluation_by_class_df = pd.DataFrame(results)

    class_list.append('None')
    confusion_matrix_df = pd.DataFrame(data=confusion_matrix, index=class_list, columns=class_list, dtype=np.int)

    return confusion_matrix_df, evaluation_by_class_df


def create(parsed_data_list, labelmap_path):
    label_map = label_map_util.load_labelmap(labelmap_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)

    confusion_matrix = process_detections(parsed_data_list, categories)

    confusion_matrix_df, evaluation_by_class_df = get_evaluations(confusion_matrix, categories)

    return confusion_matrix_df, evaluation_by_class_df


if __name__ == '__main__':
    tf.app.run(create)
