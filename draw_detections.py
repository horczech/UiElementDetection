from utilities.tf_utils import run_detector
from utilities.detection_parser import DetectionParser
from os import path
from object_detection.utils import visualization_utils as vis_util
from utils import pil_img_to_numpy_array, load_img_to_np
from object_detection.utils import label_map_util
import numpy as np
from matplotlib import pyplot as plt
import cv2
from detektor import ObjectDetector
from glob import glob

def convert_bbox_to_y_x_min_max(x,y, width, height):
    ymin=y
    xmin=x
    ymax=y+height
    xmax=x+width

    return ymin, xmin, ymax, xmax

def draw_bboxes(img_path, bbox_data, label_map_path):
    image_np = load_img_to_np(img_path)

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # converted_bbox = np.zeros_like(bbox_data.coordinates.coordinates)
    # for idx, bbox_coord in enumerate(bbox_data.coordinates.coordinates):
    #     x, y, width, height = [item for item in bbox_coord]
    #     bbox_abs = convert_bbox_to_y_x_min_max(x, y, width, height)
    #     converted_bbox[idx, :] = bbox_abs
    scores = bbox_data.scores
    if scores is None:
        scores = np.ones_like(bbox_data.class_ids, dtype=float)


    vis_util.visualize_boxes_and_labels_on_image_array(
        image=image_np,
        boxes=bbox_data.coordinates.coordinates,
        classes=bbox_data.class_ids,
        scores=scores,
        category_index=category_index,
        use_normalized_coordinates=True,
        groundtruth_box_visualization_color='Green',
        line_thickness=8)

    return image_np


def draw_results(parsed_data_list, img_dir, labelmap_path, output_dir):

    for img_id, image_data in enumerate(parsed_data_list):
        print(f'Saving image No.{img_id} out of {len(parsed_data_list)}')
        img_path = path.join(img_dir, image_data.image_name)
        save_img_path = path.join(output_dir, image_data.image_name)

        try:
            # Draw detections bboxes
            dt_img = draw_bboxes(img_path, image_data.detected_bboxes, labelmap_path)
            # Draw ground truth bboxes
            gt_img = draw_bboxes(img_path, image_data.groundtruth_bboxes, labelmap_path)


            stacked_images = np.hstack((dt_img, gt_img))
            stacked_images = cv2.cvtColor(stacked_images,cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_img_path,stacked_images)
        except Exception as e:
            print(e)
            print(f'Skipped image: {img_path}')
            continue

        # dpi = 160
        # height = image_data.image_height
        # width = image_data.image_width
        # figsize = width / float(dpi), height / float(dpi)
        # fig = plt.figure(figsize=figsize)
        # ax = fig.add_axes([0, 0, 1, 1])
        # ax.axis('off')
        # ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
        #
        # plt.subplot(121)
        # plt.imshow(dt_img)
        # plt.title('Detected objects')
        # plt.subplot(122)
        # plt.imshow(gt_img)
        # plt.title('Groundtruth')
        #
        # plt.savefig(save_img_path, dpi=dpi)


def run(input_images_record, model_path, output_dir, labelmap_path, image_dir):



    detection_record_path = run_detector(inference_graph_path=model_path,
                                         input_tfrecord_path=input_images_record,
                                         output_dir_path=output_dir)

    parser = DetectionParser(detection_record_path=detection_record_path,
                             label_map_path=labelmap_path,
                             image_dir=image_dir)

    parsed_data_list = parser.parse()

    draw_results(parsed_data_list, image_dir, labelmap_path, output_dir)



def run_blind_detector(model_path, img_dir, output_dir, label_map_path, img_suffix='.png'):

    test_image_paths = glob(path.join(img_dir, f'*{img_suffix}'))
    detector = ObjectDetector(model_path, label_map_path)

    print(f'Number of images that will be processed: {len(test_image_paths)}')
    for img_idx, img_path in enumerate(test_image_paths):
        print(f'Processing image No. {img_idx} out of {len(test_image_paths)}')
        output_filename = path.join(output_dir, path.basename(img_path))
        prediction = detector.detect_single_img(img_path)
        result_img = detector.draw_bboxes(load_img_to_np(img_path), prediction)
        cv2.imwrite(output_filename,cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # PATH_TO_IMAGE_RECORD_FILE = r'C:\Code\TFOD\test_dir\saving_images\input.records'
    # OUTPUT_DIR = r'C:\Code\TFOD\test_dir\saving_images\output_images'
    # IMAGE_DIR = r'C:\Code\TFOD\test_dir\saving_images\input_images'
    #
    # PATH_TO_MODEL = r'C:\Code\TFOD\assets\trained_models\faster_rcnn_resnet50_coco_2018_01_28_v1\transformed_model\frozen_inference_graph.pb'
    # LABEL_MAP_PATH = r'C:\Code\TFOD\assets\trained_models\faster_rcnn_resnet50_coco_2018_01_28_v1\label_map.pbtxt'

    OUTPUT_DIR = r'C:\Code\Datasets\printer\out_filter'
    IMAGE_DIR = r'C:\Code\Datasets\dummy\in'
    PATH_TO_MODEL = r"C:\Code\TFOD\assets\trained_models\faster_rcnn_resnet50_coco_2018_01_28_v1\v2\frozen_inference_graph.pb"
    LABEL_MAP_PATH = r"C:\Code\TFOD\assets\trained_models\label_map.pbtxt"
    PATH_TO_IMAGE_RECORD_FILE = r"C:\Code\Datasets\processed_datasets\checked_android\dataset_without_bullshit_listitem_classes\test.record"

    # run(input_images_record=PATH_TO_IMAGE_RECORD_FILE,
    #     model_path=PATH_TO_MODEL,
    #     output_dir=OUTPUT_DIR,
    #     labelmap_path=LABEL_MAP_PATH,
    #     image_dir=IMAGE_DIR)



    run_blind_detector(model_path=PATH_TO_MODEL,
                       img_dir=IMAGE_DIR,
                       output_dir=OUTPUT_DIR,
                       label_map_path=LABEL_MAP_PATH)
