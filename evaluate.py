from utilities.tf_utils import run_detector
from utilities import confusion_matrix
from os import path
from utilities.detection_parser import DetectionParser
from utilities.coco_utils import generate_ground_truth_coco_data, generate_detection_coco_data, evaluate_from_file
import json
import cv2
import numpy as np
from utilities.visualization.bbox_drawer import BboxDrawer
import os
from models.research.object_detection import export_inference_graph
from glob import glob


def create_confusion_matrix(parsed_data_list, labelmap_path, output_dir_path):
    print(f'Creating confusion matrix')

    CONFUSION_MATRIX_FILE_NAME = 'confusion_matrix.csv'
    CLASS_EVALUATION_FILE_NAME = 'class_evaluation.csv'

    confusion_matrix_path = path.join(output_dir_path, CONFUSION_MATRIX_FILE_NAME)
    class_evaluation_path = path.join(output_dir_path, CLASS_EVALUATION_FILE_NAME)

    confusion_matrix_df, evaluation_by_class_df = confusion_matrix.create(parsed_data_list, labelmap_path)

    confusion_matrix_df.to_csv(confusion_matrix_path)
    evaluation_by_class_df.to_csv(class_evaluation_path)

    print(f'Class evaluation:\n{evaluation_by_class_df.to_string()}\n\n'
          f'Confusion matrix:\n{confusion_matrix_df.to_string()}')

    print(f'\n\n'
          f'Confusion matrix saved to: {confusion_matrix_path}\n'
          f'Class evaluation saved to: {class_evaluation_path}')

    return confusion_matrix_df, evaluation_by_class_df


def create_coco_evaluation(parsed_data_list, labelmap_path, output_dir_path):
    print('Creating coco metrics')
    DETECTION_FILE_NAME = 'detection_coco.json'
    GROUNDTRUTH_FILE_NAME = 'groundtruth_coco.json'

    detection_file_path = path.join(output_dir_path, DETECTION_FILE_NAME)
    groundtruth_file_path = path.join(output_dir_path, GROUNDTRUTH_FILE_NAME)

    groundtruth_dict = generate_ground_truth_coco_data(parsed_data_list, labelmap_path)
    detection_dict = generate_detection_coco_data(parsed_data_list, labelmap_path)

    with open(detection_file_path, 'w') as file:
        json.dump(detection_dict, file)

    with open(groundtruth_file_path, 'w') as file:
        json.dump(groundtruth_dict, file)

    return detection_file_path, groundtruth_file_path


def run_coco_evaluation(detection_file_path, groundtruth_file_path, output_dir_path):
    print('Running COCO evaluation...')

    OUTPUT_FILE_NAME = 'coco_evaluation.txt'
    output_dir_path = path.join(output_dir_path, OUTPUT_FILE_NAME)

    report_string = evaluate_from_file(detection_file_path, groundtruth_file_path)

    with open(output_dir_path, "w") as text_file:
        text_file.write(report_string)

    print(f'\n\nCOCO evaluation saved to: {output_dir_path}')


def evaluate(model_dir_path, eval_tfrecord_path, eval_img_dir_path, labelmap_path, should_draw_results=False):


    if not path.exists(labelmap_path):
        raise ValueError(
            f"The file labelmap.pbtxt NOT FOUND. The file does not exists on path: {labelmap_path}")

    detection_record_path, evaluation_dir_path = initialize_evaluation(model_dir_path,
                                                                       eval_tfrecord_path)

    parser = DetectionParser(detection_record_path=detection_record_path,
                             label_map_path=labelmap_path,
                             image_dir=eval_img_dir_path)

    parsed_data_list = parser.parse()

    create_confusion_matrix(parsed_data_list=parsed_data_list,
                            labelmap_path=labelmap_path,
                            output_dir_path=evaluation_dir_path)

    detection_file_path, groundtruth_file_path = create_coco_evaluation(parsed_data_list=parsed_data_list,
                                                                        labelmap_path=labelmap_path,
                                                                        output_dir_path=evaluation_dir_path)

    run_coco_evaluation(detection_file_path=detection_file_path,
                        groundtruth_file_path=groundtruth_file_path,
                        output_dir_path=evaluation_dir_path)

    if should_draw_results:
        save_img_dir = path.join(evaluation_dir_path, 'images')
        if not path.exists(save_img_dir):
            os.mkdir(save_img_dir)

        draw_detections(parsed_data_list, eval_img_dir_path, labelmap_path, save_img_dir)

    print('EVALUATION SUCCESFULLY FINISHED!!')


def initialize_evaluation(dir_with_trained_ckpt, tf_records_to_evaluate):
    pipeline_path = path.join(dir_with_trained_ckpt, r'pipeline.config')
    if not path.exists(pipeline_path):
        raise ValueError(
            f"The directory must contain pipeline.config file. The file does not exists on path: {pipeline_path}")

    inference_graph_path = path.join(dir_with_trained_ckpt, 'frozen_inference_graph.pb')
    if not path.exists(inference_graph_path):
        print(
            f'INFO: frozen_inference_graph.pb NOT found so it will be generated from checpoint file. Path of frozen grah: {inference_graph_path}')
        create_frozen_inference_graph(dir_with_trained_ckpt, pipeline_path)

    evaluation_dir_path = path.join(dir_with_trained_ckpt, 'evaluation')
    if not path.exists(evaluation_dir_path):
        print(f'INFO: Creating a directory for evaluation on path {evaluation_dir_path}')
        os.mkdir(evaluation_dir_path)
    infer_detection_path = path.join(evaluation_dir_path, 'infer_detections.record')

    if not path.exists(infer_detection_path):
        print(
            f'INFO: detection data were not found so the model will be run to generate the detection data. File not found on path: {infer_detection_path}')

        detection_record_path = run_detector(inference_graph_path=inference_graph_path,
                                             input_tfrecord_path=tf_records_to_evaluate,
                                             output_dir_path=evaluation_dir_path)
    else:
        print(f'SKIPPING DETECTION PART. USING PREGENERATED INFER RECORD FILE')
        detection_record_path = infer_detection_path

    return detection_record_path, evaluation_dir_path


def create_frozen_inference_graph(dir_with_trained_ckpt, pipeline_path):
    from models.research.object_detection.export_inference_graph import FLAGS
    ckpt_file_list = glob(path.join(dir_with_trained_ckpt, r'model.ckpt-*.meta'))
    if len(ckpt_file_list) != 1:
        raise ValueError("The directory HAVE to contain only one checkpoint file.")
    ckpt_path = ckpt_file_list[0].replace('.meta', '')
    FLAGS.input_type = r'image_tensor'
    FLAGS.pipeline_config_path = pipeline_path
    FLAGS.output_directory = dir_with_trained_ckpt
    FLAGS.trained_checkpoint_prefix = ckpt_path
    export_inference_graph.main(None)

    FLAGS.remove_flag_values(FLAGS.flag_values_dict())


def draw_detections(parsed_data_list, image_dir, labelmap_path, save_img_dir):
    drawer = BboxDrawer(labelmap_path)

    for idx, data in enumerate(parsed_data_list):
        print(f'Drawing image {idx}/{len(parsed_data_list)}')
        image_path = path.join(image_dir, data.image_name)
        image = cv2.imread(image_path)

        gt_image = drawer.draw_detections(image=image.copy(),
                                          bboxes=data.groundtruth_bboxes.coordinates.coordinates,
                                          class_indexes=data.groundtruth_bboxes.class_ids,
                                          detection_scores=data.groundtruth_bboxes.scores,
                                          use_normalized_coordinates=True)

        dt_image = drawer.draw_detections(image=image.copy(),
                                          bboxes=data.detected_bboxes.coordinates.coordinates,
                                          class_indexes=data.detected_bboxes.class_ids,
                                          detection_scores=data.detected_bboxes.scores,
                                          use_normalized_coordinates=True)

        merged_images = np.hstack((dt_image, gt_image))

        save_image_path = path.join(save_img_dir, data.image_name)
        if not path.exists(path.dirname(save_image_path)):
            os.mkdir(path.dirname(save_image_path))

        cv2.imwrite(save_image_path, merged_images)


if __name__ == '__main__':
    PATH_TO_DIR = r'C:\Users\horakm\Desktop\test\test'
    TEST_DATA_RECORD = r"C:\Code\Dataset2\annotations\printer\phase_2\test_full_printer_phase2.record"
    PATH_TO_EVALUATED_IMAGE_DIR = r'C:\Code\Dataset2\images'
    LABELMAP_PATH = r"C:\Code\Dataset2\label_maps\label_map_7_classes.pbtxt"
    DRAW_RESULTS = True

    evaluate(model_dir_path=PATH_TO_DIR,
             eval_tfrecord_path=TEST_DATA_RECORD,
             eval_img_dir_path=PATH_TO_EVALUATED_IMAGE_DIR,
             labelmap_path=LABELMAP_PATH,
             should_draw_results=DRAW_RESULTS)
