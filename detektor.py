import tensorflow as tf

from PIL import Image

import numpy as np

from glob import glob
from os import path
from utils import pil_img_to_numpy_array, load_img_to_np
from matplotlib import pyplot as plt
from coco_utils import convert_to_coco_format
import json
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



class ObjectDetector:

    def __init__(self, path_to_frozen_model, path_to_labels):
        self._model_path = path_to_frozen_model
        self._lales_path = path_to_labels

        self.detection_graph = self._build_graph()
        self.sess = tf.Session(graph=self.detection_graph)

        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def _build_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def _get_image_name_from_path(self, img_path):
        return path.basename(img_path)

    def detect_single_img(self, image_path):
        image = Image.open(image_path)
        img_width, img_height = image.size

        image_np = pil_img_to_numpy_array(image)

        # REMOVE
        image_np = cv2.fastNlMeansDenoisingColored(image_np, None, 10, 10, 7, 21)

        image_np_expanded = np.expand_dims(image_np, axis=0)

        graph = self.detection_graph
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        boxes = graph.get_tensor_by_name('detection_boxes:0')
        scores = graph.get_tensor_by_name('detection_scores:0')
        classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
        boxes, scores, classes, num_detections = map(np.squeeze, [boxes, scores, classes, num_detections])

        output_dict = dict()
        output_dict['num_detections'] = int(num_detections)
        output_dict['detection_classes'] = classes.astype(np.uint8)
        output_dict['detection_boxes'] = boxes
        output_dict['detection_scores'] = scores
        output_dict['image_name'] = self._get_image_name_from_path(image_path)
        output_dict['image_width'] = img_width
        output_dict['image_height'] = img_height

        return output_dict

    def draw_bboxes(self, image_np, predictions):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            predictions['detection_boxes'],
            predictions['detection_classes'],
            predictions['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        return image_np


    # def detect_objects(image_path):
    #   image = Image.open(image_path).convert('RGB')
    #   boxes, scores, classes, num_detections = client.detect(image)
    #   image.thumbnail((480, 480), Image.ANTIALIAS)
    #
    #   new_images = {}
    #   for i in range(num_detections):
    #     if scores[i] < 0.7: continue
    #     cls = classes[i]
    #     if cls not in new_images.keys():
    #       new_images[cls] = image.copy()
    #     draw_bounding_box_on_image(new_images[cls], boxes[i],
    #                                thickness=int(scores[i]*10)-4)
    #
    #   result = {}
    #   result['original'] = encode_image(image.copy())
    #
    #   for cls, new_image in new_images.iteritems():
    #     category = client.category_index[cls]['name']
    #     result[category] = encode_image(new_image)
    #
    #   return result

if __name__ == '__main__':

    PATH_TO_FROZEN_GRAPH = r"C:\Code\TrainedModels\faster_rcnn_inception_v2_coco_2018_01_28\trained_just_on_printer_dataset\frozen_inference_graph.pb"
    PATH_TO_LABELS = r"C:\Code\TrainedModels\label_map.pbtxt"
    PATH_TO_TEST_IMAGES_DIR = r'C:\Code\Datasets\dummy\in'
    IMG_SUFFIX = '.png'

    test_image_paths = glob(path.join(PATH_TO_TEST_IMAGES_DIR, f'*{IMG_SUFFIX}'))

    detector = ObjectDetector(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)

    predictions = []
    for img_idx, img_path in enumerate(test_image_paths):
        print(f'Working on image No. {img_idx} out of {len(test_image_paths)}')

        prediction = detector.detect_single_img(img_path)
        result_img = detector.draw_bboxes(load_img_to_np(img_path), prediction)
        predictions.extend(convert_to_coco_format(prediction))

        # plt.figure()
        # plt.imshow(result_img)
        # plt.show()

    with open('test_dir/predictions.json', 'w') as file:
        json.dump(predictions, file)