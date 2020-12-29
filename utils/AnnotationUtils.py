import cv2
import glob
import io
import json
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf

import sys
sys.path.append('../../../tensorflow/models/research')

from collections import namedtuple
from object_detection.utils import dataset_util
from PIL import Image

# TODO: This should be temporary! Fix annotation files to be consistent
TRANSLATE = {"publication": "copyright",
             "copyrights": "copyright",
             "data": "list"}

def write_cropped_overlapped_images(img_out_path, mask_out_path, img_arr, mask_arr, target_size, pct_overlapp):
    total_x = int(float(len(img_arr[0]))*(1+pct_overlapp))
    total_x_overlap = total_x - len(img_arr[0])
    total_x_moves = math.ceil(float(total_x)/target_size)
    overlap_per_x_move = int(((total_x_moves*target_size) - len(img_arr[0])) / total_x_moves)
    
    x_indexes = []
    cur_x = 0
    for i in range(total_x_moves-1):
        x_indexes.append(cur_x)
        cur_x += target_size + 1 - overlap_per_x_move
    cur_x = len(img_arr[0]) - 1 - target_size  # Compensate for rounding
    x_indexes.append(cur_x)
    
    total_y = int(float(len(img_arr))*(1+pct_overlapp))
    total_y_overlap = total_y - len(img_arr)
    total_y_moves = math.ceil(float(total_y)/target_size)
    overlap_per_y_move = int(((total_y_moves*target_size) - len(img_arr)) / total_y_moves)
    
    y_indexes = []
    cur_y = 0
    for i in range(total_y_moves-1):
        y_indexes.append(cur_y)
        cur_y += target_size + 1 - overlap_per_y_move
    cur_y = len(img_arr) - 1 - target_size  # Compensate for rounding
    y_indexes.append(cur_y)
    
    i = 0
    mask_tags = {}
    for x in x_indexes:
        for y in y_indexes:
            cropped_img = img_arr[y:y+target_size,x:x+target_size]
            cropped_mask = mask_arr[y:y+target_size,x:x+target_size]
            
            img_path = img_out_path.replace('.', '-{}.'.format(i))
            mask_path = mask_out_path.replace('.', '-{}.'.format(i))
            
            cv2.imwrite(img_path, cropped_img)
            cv2.imwrite(mask_path, cropped_mask)
            
            cropped_mask = np.where(cropped_mask == 255, 0, cropped_mask)
            mask_tags[img_path] = np.unique(cropped_mask)
            i += 1
    return mask_tags

def is_inner_class(area_and_boxes):
    inner_classes = set()
    for area, x, y, w, h, rgb in area_and_boxes:
        for area_2, x_2, y_2, w_2, h_2, rgb_2 in area_and_boxes:
            if rgb[0] == rgb_2[0]:
                continue
            if area == area_2 and x == x_2 and y == y_2:
                continue
            xA = np.maximum(x, x_2)
            yA = np.maximum(y, y_2)
            xB = np.minimum(x+w, x_2+w_2)
            yB = np.minimum(y+h, y_2+h_2)
            
            # compute the area of intersection rectangle
            interArea = np.abs(np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0))
            if interArea > 0 and area_2 < area:
                inner_classes.add(rgb_2[0])
    return inner_classes
                

def write_masks_from_annotation(vott_filepath, mask_out_path, tag_names=None, 
                                split_crop=False, tag_mapping=None, 
                                buffer_size=0, force=False):
    """
    Takes in an annotation JSON from VOTT, and creates masks files 
    described by the bounding boxes and tags in the JSON. Returns a list of 
    tags found and a mapping of class->tag name, and a set of tags actually used.
    
    tag_names, class_to_tag, file_to_tags_used = write_masks_from_annotation("train/annotations/doc/doc.json", "train/masks")

    Parameters
    ----------
    vott_filepath : string
        Path to a VOTT annotation JSON
    mask_out_path : string
        Folder to place masks into
    tag_names : list
        Override the tag_names list so that the values in the json are ignored
    buffer_size : int
        The size of a buffer "don't care" region between a mask and the background. 
        The buffer is subtracted from the mask, making the mask slightly smaller
    split_crop : book
        Write out the images and masks split into overlapping tiles
    force : bool
        If true, will write files to disk if they already exist

    Returns
    -------
    tag_names, class_dict, file_to_tags_used : list, dict, dict
        A list of tags found, a dict mapping of class->tag name, and a dict mapping of img->tags actually used
    """
    vott_json = open(vott_filepath, 'r')
    dataset = json.load(vott_json)
    vott_json.close()
    
    if tag_names == None:
        tag_names = set([x["name"].lower()for x in sorted(dataset["tags"], key=lambda x: x["name"])])
    
    class_mapping = {}
    i = 1
    for key in sorted(tag_names):
        if key == 'background':
            continue
        if tag_mapping and key in tag_mapping:
            continue
        key = key.lower()
        class_mapping[key] = (i, i ,i)
        i += 1
    
    # Special case the background
    class_mapping['background'] = (0, 0, 0)
    tag_names.add('background')
    used_tags = {}
    tag_counts = {True: {}, False: {}}
    tag_areas = {True: {}, False: {}}
    for ano_id, ano_data in dataset['assets'].items():
        img_index = ano_data['asset']['path'].split('-')[1].split('.')[0].lstrip('0')
        img_name = os.path.basename(vott_filepath).replace("json", "jpg")
        img_dir = os.path.join(os.path.dirname(vott_filepath.replace('annotations', 'documents')),
                               img_name.split('.')[0])
        img_name = img_name.replace('.jpg', '-{}.jpg'.format(img_index))
        img_path = os.path.join(img_dir, img_name)
        
        # Special case. Sometimes img's have a leading zero in vol num, sometimes not
        if not os.path.exists(img_path):
            img_name = img_name.replace("-{}.jpg".format(img_index), "-0{}.jpg".format(img_index))
            img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            print(img_path)
        
        first_page = False
        first_pages = list(filter(lambda x: "txt" not in x, os.listdir(img_dir)))
        if img_name in first_pages[:2]:
            first_page = True
        
        box_path = img_path.replace(".jpg", ".txt")
        
        img = cv2.imread(img_path)
        img = np.zeros(img.shape)
        
        area_and_boxes = []
        used_tags[img_path] = set()
        for region in ano_data["regions"]:
            key = region["tags"][-1].lower()
            if key not in class_mapping:
                continue
            
            if key in TRANSLATE:
                key = TRANSLATE[key]
            
            if tag_mapping and key in tag_mapping:
                key = tag_mapping[key]
                
            if not split_crop:
                used_tags[img_path].add(key)
                if key not in tag_counts[first_page]:
                    tag_counts[first_page][key] = 0
                    tag_areas[first_page][key] = 0
                tag_counts[first_page][key] += 1
            
            rgb = class_mapping[key]
            x = int(region["boundingBox"]["left"])
            y = int(region["boundingBox"]["top"])
            w = int(region["boundingBox"]["width"])
            h = int(region["boundingBox"]["height"])
            area = w*h
            tag_areas[first_page][key] += area
            area_and_boxes.append((area, x, y, w, h, rgb))
        
        # sort by area, so smaller boxes are on the top layer, also write out box file
        if os.path.exists(box_path):
            os.remove(box_path)
        
        area_and_boxes.sort(reverse=True)
        box_f = open(box_path, 'w+')
        for area, x, y, w, h, rgb in area_and_boxes:
            if buffer_size > 0:
                cv2.rectangle(img, (x-buffer_size,y-buffer_size), (x+w+buffer_size, y+h+buffer_size), (255, 255, 255), -1)
            cv2.rectangle(img, (x+buffer_size, y+buffer_size), (x+w-buffer_size, y+h-buffer_size), rgb, -1)
            box_f.write("{},{},{},{},{}\n".format(rgb[0], x, y, w, h))
        box_f.close()
        
                
        if not os.path.exists(mask_out_path):
            os.mkdir(mask_out_path)
        
        mask_dir = os.path.join(mask_out_path, os.path.basename(img_dir))
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
    
        outfile = os.path.join(mask_dir, os.path.basename(img_path).replace('jpg', 'png'))
        
        if split_crop:
            cropped_mask_path = os.path.join(os.path.dirname(outfile), 'cropped')
            if not os.path.exists(cropped_mask_path):
                os.mkdir(cropped_mask_path)
            cropped_mask_path = os.path.join(cropped_mask_path, os.path.basename(img_path).replace('jpg', 'png'))
            
            cropped_img_path = os.path.join(os.path.dirname(img_path), 'cropped')
            if not os.path.exists(cropped_img_path):
                os.mkdir(cropped_img_path)
            cropped_img_path = os.path.join(cropped_img_path, os.path.basename(img_path))
            
            mask_tags = write_cropped_overlapped_images(cropped_img_path, cropped_mask_path, cv2.imread(img_path), img, 672, 0.1)
            used_tags.update(mask_tags)
        else:
            if not os.path.exists(outfile) or force:
                cv2.imwrite(outfile, img)
    
    actual_class_mapping = {}
    for key in sorted(class_mapping):
        key = key.lower()
        actual_class_mapping[class_mapping[key][0]] = key
    
    inner_classes = is_inner_class(area_and_boxes)
    
    return tag_names, actual_class_mapping, used_tags, tag_counts, tag_areas, inner_classes

def apply_mask_localization(original_image, mask_image, mask_class):
    """
    Takes in a path to an image, path to corresponding mask image array, and 
    the mask class number to apply from the mask image, and return the new localized 
    image according to the mask. This assumes the mask image is made up of 
    multiple masks.

    localized_image = apply_mask_localization(orig_img, img_mask, class_name_to_class_number["core_text"])

    Parameters
    ----------
    original_image : string
        The image you want to apply a mask to
    mask_image : string
        The mask image to select a mask from.
    mask_class : int
        The class number to select from the mask

    Returns
    -------
    result : 3D numpy array
        Reurns the resulting image under the chosen mask

    """
    if type(original_image) == str:
        original_image = cv2.imread(original_image)
    
    if type(mask_image) == str:
        mask_image = cv2.imread(mask_image)

    mask_lower_bound = np.array([mask_class, 0, 0])
    mask_upper_bound = np.array([mask_class, mask_class, mask_class])
    
    mask = cv2.inRange(mask_image, mask_lower_bound, mask_upper_bound)
    result = cv2.bitwise_and(original_image, original_image, mask=mask)
    
    return result

""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
"""

def xml_to_csv(paths, mask_out_path, tag_names, 
               split_crop=False, tag_mapping=None, 
               buffer_size=0, force=False):
    """
    Takes in an annotation JSON from VOTT, and creates masks files 
    described by the bounding boxes and tags in the JSON. Returns a list of 
    tags found and a mapping of class->tag name, and a set of tags actually used.
    
    tag_names, class_to_tag, file_to_tags_used = write_masks_from_annotation("train/annotations/doc/doc.json", "train/masks")

    Parameters
    ----------
    vott_filepath : string
        Path to a VOTT annotation JSON
    mask_out_path : string
        Folder to place masks into
    tag_names : list
        Override the tag_names list so that the values in the json are ignored
    buffer_size : int
        The size of a buffer "don't care" region between a mask and the background. 
        The buffer is subtracted from the mask, making the mask slightly smaller
    split_crop : book
        Write out the images and masks split into overlapping tiles
    force : bool
        If true, will write files to disk if they already exist

    Returns
    -------
    tag_names, class_dict, file_to_tags_used : list, dict, dict
        A list of tags found, a dict mapping of class->tag name, and a dict mapping of img->tags actually used
    """   
    class_mapping = {}
    i = 1
    for key in sorted(tag_names):
        if key == 'background':
            continue
        if tag_mapping and key in tag_mapping:
            continue
        key = key.lower()
        class_mapping[key] = (i, i ,i)
        i += 1
    
    # Special case the background
    class_mapping['background'] = (0, 0, 0)
    tag_names.add('background')
    used_tags = {}
    
    object_list = []
    for vott_filepath in paths:
         vott_json = open(vott_filepath, 'r')
         dataset = json.load(vott_json)
         vott_json.close()
         for ano_id, ano_data in dataset['assets'].items():
            img_index = ano_data['asset']['path'].split('-')[1].split('.')[0].lstrip('0')
            img_name = os.path.basename(vott_filepath).replace("json", "jpg")
            img_dir = os.path.join(os.path.dirname(vott_filepath.replace('annotations', 'documents')),
                                   img_name.split('.')[0])
            img_name = img_name.replace('.jpg', '-{}.jpg'.format(img_index))
            img_path = os.path.join(img_dir, img_name)
            
            # Special case. Sometimes img's have a leading zero in vol num, sometimes not
            if not os.path.exists(img_path):
                img_name = img_name.replace("-{}.jpg".format(img_index), "-0{}.jpg".format(img_index))
                img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                print(img_path)
            
            area_and_boxes = []
            used_tags[img_path] = set()
            for region in ano_data["regions"]:
                key = region["tags"][0].lower()
                if key not in class_mapping:
                    continue
                
                if key in TRANSLATE:
                    key = TRANSLATE[key]
                
                if tag_mapping and key in tag_mapping:
                    key = tag_mapping[key]
                    
                if not split_crop:
                    used_tags[img_path].add(key)
                
                class_number = class_mapping[key]
                class_name = key
                x = int(region["boundingBox"]["left"])
                y = int(region["boundingBox"]["top"])
                w = int(region["boundingBox"]["width"])
                h = int(region["boundingBox"]["height"])
                value = (img_path,
                         w,
                         h,
                         class_name,
                         x,
                         y,
                         x+w,
                         y+h)
                object_list.append(value)

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    object_df = pd.DataFrame(object_list, columns=column_name)
    return object_df, class_mapping

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path, class_mapping):
    with tf.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
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
        classes.append(class_mapping[row['class']])

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

def write_object_tfrecords(paths):
    writer = tf.python_io.TFRecordWriter(args.output_path)
    examples, class_mapping = xml_to_csv(paths)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, class_mapping)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
