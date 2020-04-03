import csv
import json


CLASS_DICT = {
    'StaticText': 1,
    'EditText': 2,
    'ImageButton': 3,
    'RadioButton': 4,
    'Switch': 5,
    'CheckBox': 6,
    'Button': 7,
    'StaticImage': 8,
    'ListItem': 9
}


def convert_class_name_to_class_id(class_name):
    return CLASS_DICT[class_name]



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



def convert_absolute_bbox_annotation(xmin, ymin, xmax, ymax):

    bbox_top_left_x = xmin
    bbox_top_left_y = ymin
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    return bbox_top_left_x, bbox_top_left_y, bbox_width, bbox_height


def convert_to_coco_format(prediction_dict):
    img_name = prediction_dict['image_name'].split('.')[0]

    try:
        img_id = int(img_name)
    except ValueError:
        raise ValueError('Image names have to be numerical in order to convert them to image ID')

    predictions = []
    for prediction_idx in range(prediction_dict['num_detections']):
        category_id = int(prediction_dict['detection_classes'][prediction_idx])
        bbox = convert_normalized_coordinates(prediction_dict['detection_boxes'][prediction_idx],
                                              prediction_dict['image_width'],
                                              prediction_dict['image_height'])
        score = float(prediction_dict['detection_scores'][prediction_idx])

        coco_format_prediction = {'image_id': img_id,
                                   'category_id': category_id,
                                   'bbox': bbox,
                                   'score': score}
        predictions.append(coco_format_prediction)

    return predictions


def convert_csv_to_coco_format(csv_file_path, output_file_path):
    info = {
            "description": "dummy description",
            "url": "dummy url",
            "version": "11.1",
            "year": 2022,
            "contributor": "Martin Horak",
            "date_created": "today"
    }

    licences = [
        {
        "url": "dummy url",
        "id": 1,
        "name": "dummy name"
        }
    ]

    categories = [
        {"supercategory": "UiElement", "id": 1, "name": "StaticText"},
        {"supercategory": "UiElement", "id": 2, "name": "EditText"},
        {"supercategory": "UiElement", "id": 3, "name": "ImageButton"},
        {"supercategory": "UiElement", "id": 4, "name": "RadioButton"},
        {"supercategory": "UiElement", "id": 5, "name": "Switch"},
        {"supercategory": "UiElement", "id": 6, "name": "CheckBox"},
        {"supercategory": "UiElement", "id": 7, "name": "Button"},
        {"supercategory": "UiElement", "id": 8, "name": "StaticImage"},
        {"supercategory": "UiElement", "id": 9, "name": "ListItem"},
    ]


    annotations = []
    images = []


    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)

        for annotation_id, row in enumerate(csv_reader):
            print(f'Working on annotation No.: {annotation_id}')
            filename, width, height, class_name, xmin, ymin, xmax, ymax = row

            if filename == 'filename':
                continue

            try:
                image_id = int(filename.split('.')[0])
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                width = int(width)
                height = int(height)
            except ValueError:
                raise ValueError("Name of the image have to be numerical")

            dummy_segmentation_data = [666.0, 666.0, 666.0, 666.0, 666.0, 666.0]
            area = (xmax-xmin) * (ymax-ymin)
            iscrowd = 0
            bbox = convert_absolute_bbox_annotation(xmin, ymin, xmax, ymax)
            category_id = convert_class_name_to_class_id(class_name)

            image = {
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id
            }

            annotation = {
                # "segmentation": [dummy_segmentation_data],
                "area": area,
                "iscrowd": iscrowd,
                "image_id": image_id,
                "bbox": bbox,
                "category_id": category_id,
                "id": annotation_id
            }

            annotations.append(annotation)
            images.append(image)

    coco_annotation = {
        "info": info,
        "licenses": licences,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


    with open(output_file_path, 'w') as outfile:
        json.dump(coco_annotation, outfile)

    print(">>>> Successfully DONE")


if __name__ == '__main__':
    CSV_FILE_PATH = r'C:\Code\TFOD\assets\android_dataset\raw_dataset\test_labels.csv'
    OUTPUT_JSON_PATH = r'C:\Code\TFOD\test_dir\groundtruth_annotaions.json'

    convert_csv_to_coco_format(CSV_FILE_PATH, OUTPUT_JSON_PATH)