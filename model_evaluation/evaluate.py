from utilities.tf_utils import run_detector
from model_evaluation import confusion_matrix
from os import path
from utilities.detection_parser import DetectionParser
from utilities.coco_utils import generate_ground_truth_coco_data, generate_detection_coco_data, evaluate_from_file
import json


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


def evaluate(model_path, labelmap_path, tf_records_to_evaluate, test_img_dir, output_dir_path, infer_detection_path=None):

    if infer_detection_path is None:
        detection_record_path = run_detector(inference_graph_path=model_path,
                                             input_tfrecord_path=tf_records_to_evaluate,
                                             output_dir_path=output_dir_path)
    else:
        print(f'SKIPPING DETECTION PART. USING PREGENERATED INFER RECORD FILE')
        detection_record_path = infer_detection_path

    parser = DetectionParser(detection_record_path=detection_record_path,
                             label_map_path=labelmap_path,
                             image_dir=test_img_dir)

    parsed_data_list = parser.parse()

    create_confusion_matrix(parsed_data_list=parsed_data_list,
                            labelmap_path=labelmap_path,
                            output_dir_path=output_dir_path)

    detection_file_path, groundtruth_file_path = create_coco_evaluation(parsed_data_list=parsed_data_list,
                                                                        labelmap_path=labelmap_path,
                                                                        output_dir_path=output_dir_path)

    run_coco_evaluation(detection_file_path=detection_file_path,
                        groundtruth_file_path=groundtruth_file_path,
                        output_dir_path=output_dir_path)

    print('EVALUATION SUCCESFULLY FINISHED!!')


def draw_detections():
    pass


if __name__ == '__main__':
    # PATH_TO_MODEL = r'C:\Code\TFOD\assets\trained_models\faster_rcnn_resnet50_coco_2018_01_28_v1\transformed_model\frozen_inference_graph.pb'
    # PATH_TO_LABELMAP = r'C:\Code\TFOD\assets\trained_models\faster_rcnn_resnet50_coco_2018_01_28_v1\label_map.pbtxt'
    # TEST_DATA_RECORD = r'C:\Code\TFOD\test_dir\tfrecord_generator_output\train.record'
    # IMG_SUFFIX = r'.jpg'
    # MAX_IMAGE_COUNT = None
    # PATH_TO_RESULT_IMAGE_DIR = r'C:\Code\TFOD\test_dir\evaluation_results22'
    # PATH_TO_EVALUATED_IMAGE_DIR = r'C:\Code\TFOD\assets\android_dataset\checked_dataset\images'

    PATH_TO_MODEL = r"C:\Code\TrainedModels\faster_rcnn_inception_v2_coco_2018_01_28\only_android_dataset\fixed_600_960\frozen_inference_graph.pb"
    PATH_TO_LABELMAP = r"C:\Code\Dataset2\label_maps\label_map_8_classes.pbtxt"
    TEST_DATA_RECORD = r"C:\Code\Datasets\android_dataset\checked_android\dataset_without_bullshit_listitem_classes\test.record"
    IMG_SUFFIX = r'.png'
    MAX_IMAGE_COUNT = None
    PATH_TO_RESULT_IMAGE_DIR = r"C:\Code\TrainedModels\faster_rcnn_inception_v2_coco_2018_01_28\only_android_dataset\fixed_600_960\evaluation"
    PATH_TO_EVALUATED_IMAGE_DIR = r"C:\Code\Dataset2\images\android"
    # INFER_DETECTIONS_path = r"C:\Code\TrainedModels\faster_rcnn_inception_v2_coco_2018_01_28\trained_just_on_printer_dataset\evaluation_result\infer_detections.record"
    INFER_DETECTIONS_path = None

    evaluate(model_path=PATH_TO_MODEL,
             labelmap_path=PATH_TO_LABELMAP,
             tf_records_to_evaluate=TEST_DATA_RECORD,
             output_dir_path=PATH_TO_RESULT_IMAGE_DIR,
             test_img_dir=PATH_TO_EVALUATED_IMAGE_DIR,
             infer_detection_path=INFER_DETECTIONS_path)

    # from object_detection.inference import infer_detections
    # import tensorflow as tf
    # from object_detection.inference.infer_detections import FLAGS
    # from model_evaluation import confusion_matrix
    #
