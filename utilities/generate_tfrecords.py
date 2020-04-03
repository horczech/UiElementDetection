from glob import glob
from os import path
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
import os
import io

from PIL import Image
import random
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from object_detection.utils import label_map_util
from utilities.general_utils import get_paths_of_files_with_suffix

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util


def split_list(input_list, split_factor):
    random.shuffle(input_list)
    split_threshold = int(len(input_list)*split_factor)

    train = input_list[:split_threshold]
    test = input_list[split_threshold:]

    return train, test


def load_annotations(annotation_dir_path):
    annotation_paths = get_paths_of_files_with_suffix(r'C:\Code\Dataset2\annotations\printer\splitted\train', '*.xml')
    annotation_paths.extend(get_paths_of_files_with_suffix(r'C:\Code\Dataset2\annotations\android\checked_dataset\all', '*.xml'))


    # annotation_paths = get_paths_of_files_with_suffix(annotation_dir_path, '*.xml')

    xml_list = []
    for file_idx, ann_file in enumerate(annotation_paths):
        print(f'Loading file No.{file_idx}')


        tree = ET.parse(ann_file)
        root = tree.getroot()
        if len(root.findall('object')) == 0:
            print(f'Empty XML file. Path: {ann_file}')

        for member in root.findall('object'):
            filename = root.find('filename').text.replace('/','\\')
            bbox = member.find('bndbox')
            value = (filename,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member.find('name').text,
                     int(float(bbox.find('xmin').text)),
                     int(float(bbox.find('ymin').text)),
                     int(float(bbox.find('xmax').text)),
                     int(float(bbox.find('ymax').text))
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    xml_df = pd.DataFrame(xml_list, columns=column_name)

    return xml_df

def create_tf_example(group, path, labelmap_dict):
    # file_path = os.path.join(path, '{}.png'.format(group.filename))
    file_path = os.path.join(path, group.filename)

    if not os.path.isfile(file_path):
        print(f'WARNING: Image not found. Skipping. Path: {file_path}')
        return None

    with tf.gfile.GFile(file_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size


    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(labelmap_dict[row['class']])
        # classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def generate_tfrecords(annotation_dir_path, image_dir_path, labelmap_path, output_file, max_count=None):

    writer = tf.python_io.TFRecordWriter(output_file)
    examples = load_annotations(annotation_dir_path=annotation_dir_path)
    grouped = split(examples, 'filename')
    labelmap = label_map_util.load_labelmap(labelmap_path)
    labelmap_dict = label_map_util.get_label_map_dict(labelmap)

    for idx, group in enumerate(grouped):
        print(f'Working on file No.: {idx}/{len(grouped)}')
        tf_example = create_tf_example(group, image_dir_path, labelmap_dict)
        if tf_example is None:
            continue
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_file))


if __name__ == '__main__':
    # ANNOTATION_DIR_PATH = r'C:\Code\TFOD\assets\android_dataset\checked_dataset\annotations'
    # IMAGE_DIR_PATH = r'C:\Code\TFOD\assets\android_dataset\checked_dataset\images'
    # LABELMAP_PATH = r'C:\Code\TFOD\assets\android_dataset\label_map.pbtxt'
    # OUTPUT_FILE = r'C:\Code\TFOD\test_dir\tfrecord_generator_output\train.record'


    IMAGE_DIR_PATH = r"C:\Code\Dataset2\images"
    LABELMAP_PATH = r"C:\Code\Dataset2\label_maps\label_map_8_classes.pbtxt"

    ANNOTATION_DIR_PATH = r"C:\Code\Dataset2\annotations\android\checked_dataset\all"
    OUTPUT_FILE = r"C:\Code\Dataset2\tf_records\mixed\train_mixed_normalized_ratio_8classes.record"


    generate_tfrecords(annotation_dir_path=ANNOTATION_DIR_PATH,
                       image_dir_path=IMAGE_DIR_PATH,
                       labelmap_path=LABELMAP_PATH,
                       output_file=OUTPUT_FILE,)





