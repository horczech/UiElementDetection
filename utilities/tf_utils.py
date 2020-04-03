from os import path

from object_detection.inference import infer_detections


def run_detector(inference_graph_path, input_tfrecord_path, output_dir_path):
    print(f'Running object detection with:\n'
          f'\t Model: {inference_graph_path}\n'
          f'\t Input data: {input_tfrecord_path}\n'
          f'\t Saving output to: {output_dir_path}\n')

    from object_detection.inference.infer_detections import FLAGS
    FILE_NAME = 'infer_detections.record'

    output_file_path = path.join(output_dir_path, FILE_NAME)

    FLAGS.input_tfrecord_paths = input_tfrecord_path
    FLAGS.output_tfrecord_path = output_file_path
    FLAGS.inference_graph = inference_graph_path

    infer_detections.main(0)

    print(f'Successfully finished object detection.\n'
          f'Saving result to: {output_file_path}')
    return output_file_path