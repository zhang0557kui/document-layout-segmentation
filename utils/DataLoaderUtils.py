import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

NUM_CLASSES = tf.cast(22, tf.uint8)

def stratify_train_test_split(file_to_used_tags, test_pct, seed=42, debug=False):
    """
    Attempts to approximate stratified sampling, where both the train and test
    sets have similar distributions of classes. Setting debug=True will confirm
    if this worked. Return two lists of file paths for testing and training.

    Parameters
    ----------
    file_to_used_tags : dict
        A dictionary of all image files to tags used. This can be obtained from
        the output of write_masks_from_annotation in AnnotationUtils.py
    test_pct : float
        A number [0, 1] representing the percentage of the dataset to save as 
        test data
    seed : int, optional
        Seeds the random shuffling of data while calculating. The default is 42.
    debug : bool, optional
        If true, will show plots of distributions for both test and train. 
        The default is False.

    Returns
    -------
    train_paths : list
        List of paths to images to use in the training set.
    test_paths : list
        List of paths to images to use in the testing set.

    """
    usage_numbers = {}
    total_num_tags = 0
    for paper, tags in file_to_used_tags.items():
        for tag in tags:
            if tag not in usage_numbers:
                usage_numbers[tag] = 0
            usage_numbers[tag] += 1
            total_num_tags += 1
    
    target_usage_percentages = {}
    for tag, num in usage_numbers.items():
         target_usage_percentages[tag] = float(num)/total_num_tags
    
    train_usage_numbers = copy.deepcopy(usage_numbers)     
    test_usage_numbers = {}
    for tag, num in usage_numbers.items():
        test_usage_numbers[tag] = 0
    
    train_paths = list(file_to_used_tags.keys())
    test_paths = list()
    
    random.seed(seed)  # Doesn't matter what the seed is really, we will shuffle this later
    total_num_test_tags = 0
    total_num_train_tags = total_num_tags
    
    train_indicies = np.arange(len(train_paths))
    random.shuffle(train_indicies)
    i = 0
    iterations = 0
    target_length = len(train_paths)*test_pct
    while len(test_paths) < target_length:
        rnd_path = train_paths[train_indicies[i]]
        i += 1
        choose_path = True
        for tag in file_to_used_tags[rnd_path]:
            if iterations < len(file_to_used_tags)*0.02:
                continue
            if (float(test_usage_numbers[tag] + 1) / total_num_test_tags) > target_usage_percentages[tag]*1.5:
                choose_path = False
                break
        
        if choose_path:
            for tag in file_to_used_tags[rnd_path]:
                train_usage_numbers[tag] -= 1
                total_num_train_tags -= 1
                test_usage_numbers[tag] += 1
                total_num_test_tags += 1
                
            test_paths.append(rnd_path)
            train_paths.remove(rnd_path)
            
            train_indicies = np.arange(len(train_paths))
            random.shuffle(train_indicies)
            i = 0
            iterations += 1
        
        if i == len(train_paths):
            i = 0
            iterations += 1
    
    if debug:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.title('Data Frequency Test')
        plt.xticks(rotation=90)
        plt.bar(test_usage_numbers.keys(), test_usage_numbers.values(), 0.5, color='g')
        plt.show()
        
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.title('Data Frequency Train')
        plt.xticks(rotation=90)
        plt.bar(train_usage_numbers.keys(), train_usage_numbers.values(), 0.5, color='r')
        plt.show()
    
    return train_paths, test_paths


# Borrowed mostly from https://github.com/dhassault/tf-semantic-example/blob/master/01_semantic_segmentation_basic.ipynb
@tf.function(experimental_relax_shapes=True)
def parse_image(img_path: str, unused_classes: list, background_class: int, adjacent_mask_dir: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    unused_class : list
        A list of class integers that will be ignored and filtered out of the ground truth masks.
    
    background_class : int
        The class the represents the background. It is usually zero.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    #image = tf.image.convert_image_dtype(image, tf.int32)

    # For one Image path:
    # .../data/documents/<doc_name>/<doc_image>.jpg
    # Its corresponding annotation path is:
    # .../data/annotations/<doc_name>/<doc_image>.png
    mask_path = tf.strings.regex_replace(img_path, "documents", adjacent_mask_dir)
    #mask_path = tf.strings.regex_replace(img_path, "publaynet", adjacent_mask_dir)
    data_path = tf.strings.regex_replace(img_path, "jpg", "txt")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    def parse_data_file(filename):
        data = []
        f = open(filename.numpy().decode(), 'r')
        for line in f:
            line = line.strip().split(",")
            if len(line) == 5:
                x, y, w, h = line[1:]
                x_min = float(x)
                y_min = float(y)
                x_max = x_min + float(w)
                y_max = y_min + float(h)
                data.append(np.array([y_min, x_min, y_max, x_max]))
        f.close()
        return np.array(data)
    
    data = tf.py_function(parse_data_file, [data_path], [tf.float32])[0]
    
    for mask_num in unused_classes:
        mask = tf.where(mask == mask_num, np.dtype('int32').type(background_class), mask)
    
    return {'image': image, 'segmentation_mask': mask, 'data': data}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally. Also fixes resize errors in masks.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.math.divide(tf.cast(input_image, tf.float32), 255.0)
    
    # Fix resize errors in don't care borders
    #input_mask = tf.where(tf.math.floormod(input_mask, 1) != 0, 255.0, input_mask)
    input_mask = tf.where(input_mask > NUM_CLASSES, tf.cast(255, tf.uint8), input_mask)
    
    return input_image, input_mask

def check_cropped_objects(mask, crop_x, crop_y, crop_size, data):
    mask = np.array(mask)
    crop_x = tf.cast(crop_x, tf.int32)
    crop_y = tf.cast(crop_y, tf.int32)
    crop_size = tf.cast(crop_size, tf.int32)
    for (class_num, x, y, w, h) in data[0]:
        # determine the (x, y)-coordinates of the intersection rectangle
        x = tf.strings.to_number(x, tf.int32)
        y = tf.strings.to_number(y, tf.int32)
        w = tf.strings.to_number(w, tf.int32)
        h = tf.strings.to_number(h, tf.int32)
        xA = tf.math.maximum(x, crop_x)
        yA = tf.math.maximum(y, crop_y)
        xB = tf.math.minimum(x+w, crop_x+crop_size)
        yB = tf.math.minimum(y+h, crop_y+crop_size)
        
        # compute the area of intersection rectangle
        area = tf.multiply(w, h)
        interArea = tf.math.abs(tf.math.maximum(xB - xA, 0) * tf.math.maximum(yB - yA, 0))
        if tf.equal(interArea, 0):
            continue
        if tf.less_equal(tf.math.divide(interArea, area), 0.6):
            mask[yA:yB, xA:xB] = 255
        
    
    return mask

def scale_normalize_boxes(boxes, img_width, img_height, target_size):
    gt_boxes = np.array(boxes)
    x_scale = target_size / img_width
    y_scale = target_size / img_height
    gt_boxes[:, 0] = (gt_boxes[:, 0] * y_scale) / float(target_size)
    gt_boxes[:, 1] = (gt_boxes[:, 1] * x_scale) / float(target_size)
    gt_boxes[:, 2] = (gt_boxes[:, 2] * y_scale) / float(target_size)
    gt_boxes[:, 3] = (gt_boxes[:, 3] * x_scale) / float(target_size)
    
    return gt_boxes

@tf.function
def load_image_train(datapoint: dict, img_size: int, crop_data: bool) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.
    img_size : int
        Integer value to resize the image into
    crop_data : bool
        Boolean value indicating whether to crop images or not
    
    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']
    gt_boxes = datapoint['data']
    # data = datapoint['data']
    
    # if tf.equal(crop_data, True):
    #     if tf.equal(tf.cond(tf.greater(tf.random.uniform(()), 0.5), lambda: 1, lambda: 0), 0):
    #         iters = 0
    #         while tf.equal(0, 0):
    #             if tf.greater_equal(iters, 10):
    #                 break
                
    #             CROP_SIZE = tf.random.uniform(shape=[1,], minval=500, maxval=800, dtype=tf.int32)[0]
    #             max_x = tf.shape(input_image)[1]-CROP_SIZE-10
    #             max_y = tf.shape(input_image)[0]-CROP_SIZE-10
    #             #CROP_SIZE = 900
    #             crop_x = tf.random.uniform(shape=[1,], minval=10, maxval=max_x, dtype=tf.int32)[0]
    #             crop_y = tf.random.uniform(shape=[1,], minval=10, maxval=max_y, dtype=tf.int32)[0]
    #             upper_y = CROP_SIZE + crop_y
    #             upper_x = CROP_SIZE + crop_x
    
    #             # Remove objects that are more than 50% cropped out
    #             cropped_input_mask = tf.py_function(check_cropped_objects, [input_mask, crop_x, crop_y, CROP_SIZE, data], [tf.uint8])
    #             cropped_input_mask = tf.convert_to_tensor(cropped_input_mask[0], dtype=tf.uint8)
                
    #             # Crop the mask and image
    #             cropped_input_image = input_image[crop_y:upper_y, crop_x:upper_x, :]
    #             cropped_input_mask = cropped_input_mask[crop_y:upper_y, crop_x:upper_x, :]
    #             cropped_input_mask.set_shape((None, None, 1))
                
    #             y, _ = tf.unique(tf.keras.backend.flatten(cropped_input_mask))
    #             if tf.greater_equal(tf.size(y), 3):
    #                 input_mask = cropped_input_mask
    #                 input_image = cropped_input_image
    #                 break
    #             iters = tf.math.add(iters, 1)
    
    # Resize and normalize boxes
    gt_boxes = tf.py_function(scale_normalize_boxes, [gt_boxes, tf.shape(input_image)[1], tf.shape(input_image)[0], img_size], [tf.float32])[0]
    
    # Resize images
    input_image = tf.image.resize(input_image, (img_size, img_size))
    input_mask = tf.image.resize(input_mask, (img_size, img_size), method='nearest')    
    
    # Randomly flip images and boxes
    if tf.math.equal(tf.cond(tf.math.greater_equal(tf.random.uniform(()), 0.5), lambda: 1, lambda: 0), 0):
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
        
        
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask, gt_boxes

@tf.function
def load_image_test(datapoint: dict, img_size: int) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.
    img_size : int
        Integer value to resize the image into

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (img_size, img_size))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (img_size, img_size), method='nearest')
    gt_boxes = datapoint['data']
    
    # Resize and normalize boxes
    gt_boxes = tf.py_function(scale_normalize_boxes, [gt_boxes, tf.shape(input_image)[1], tf.shape(input_image)[0], img_size], [tf.float32])[0]
    
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask, gt_boxes
