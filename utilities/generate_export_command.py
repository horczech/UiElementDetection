import pyperclip
import sys
import os
from os import path

def run(config_file_path,trained_checkpoint_path,output_dir):
    command = f'python export_inference_graph.py ' \
              f'--input_type image_tensor ' \
              f'--pipeline_config_path {config_file_path} ' \
              f'--trained_checkpoint_prefix {trained_checkpoint_path} ' \
              f'--output_directory {output_dir}'


    pyperclip.copy(command)

    print(f'\n\nRun following commnad to export model:\n {command}')

    print(f'\n\nRun the command from:\n {os.path.join(os.getcwd(), "models", "research", "object_detection")}')


    python_paths = r'set PYTHONPATH=C:\Code\TFOD\models;C:\Code\TFOD\models\research;C:\Code\TFOD\models\research\slim'
    print(f'\n\nAdd following paths:\n {python_paths}')


if __name__ == '__main__':
    # config_file_path = r'C:\Code\TFOD\assets\trained_models\faster_rcnn_resnet50_coco_2018_01_28_v1\faster_rcnn_resnet50_coco_2018_01_28_v1.config'
    # trained_checkpoint_path = r'C:\Code\TFOD\assets\trained_models\faster_rcnn_resnet50_coco_2018_01_28_v1\model.ckpt-200000'
    # output_dir = r'C:\Code\TFOD\assets\trained_models\faster_rcnn_resnet50_coco_2018_01_28_v1\transformed_model'

    # config_file_path = r"C:\Code\TrainedModels\faster_rcnn_inception_v2_coco_2018_01_28\trained_just_on_printer_dataset\checkpoint\pipeline_printer_alone.config"
    # trained_checkpoint_path = r"C:\Code\TrainedModels\faster_rcnn_inception_v2_coco_2018_01_28\trained_just_on_printer_dataset\checkpoint\model.ckpt-17559"
    # output_dir = r"C:\Code\TrainedModels\faster_rcnn_inception_v2_coco_2018_01_28\trained_just_on_printer_dataset"


    folder = r'C:\Users\robot\Desktop\to_download\ssd_mobilenet_v2_coco_2018_03_29\printer\fixed_300_300_batch_16'
    model_name = r'model.ckpt-12811'

    config_file_path = path.join(folder, 'pipeline.config')
    trained_checkpoint_path = path.join(folder, model_name)
    output_dir = folder

    run(config_file_path=config_file_path,
        trained_checkpoint_path=trained_checkpoint_path,
        output_dir=output_dir)