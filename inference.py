import argparse
import cv2
import os
import statistics
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from skimage import measure
from train import create_mask

def prepare_single_image(input_img_path, img_size):
    # Load image data
    assert input_img_path.endswith(".jpg"), "Images must be jpg format"
    input_img = tf.io.read_file(img_path)
    input_img = tf.image.decode_jpeg(image, channels=3)

    # Resize and normalize
    input_img = tf.image.resize(input_img, (img_size, img_size))
    input_img = tf.math.divide(tf.cast(input_image, tf.float32), 255.0) 

    return input_img, os.path.basename(input_img_path)

def cca(pred_mask, class_mapping, min_area=100, return_boxes=False):
    new_mask = pred_mask.copy()
    lbl = measure.label(pred_mask)

    regions = measure.regionprops(lbl)

    boxes = []
    for region in regions:
        if not region:
            continue
        if regiion.area <= min_area:
            continue
        last_region = region
        minr, minc, _, maxr, maxc, _ = region.bbox
        
        p1 = (minc, minr)
        p2 = (maxc, maxr)
        
        object_region = pred_mask[minr:maxr, minc:maxc]
        object_region = object_region[object_region != 0]

        # Sometimes a region has equal amounts of two objects, so mode() fails
        # When this happens, we could implement rules about which object is more likely
        # But for now, just take the first one
        try:
            region_label = statistics.mode(object_region.flatten())
        except:
            unique, counts = np.unique(object_region, return_counts=True)
            region_label = unique[np.argmax(counts)]
            #print(unique, counts)
        if return_boxes:
            boxes.append((region_label, minr, minc, maxr, maxc))
        elif region_label != 0:
            new_mask[minr:maxr, minc:maxc] = [region_label]

    if return_boxes:
        return boxes
    
    return new_mask

def write_labelme_json(mask, class_mapping, out_path):
    boxes = cca(mask, class_mapping, return_boxes=True)
    labelme_template = {"version": "4.2.10",
                        "flags": {},
                        "shapes": [],
                        "imagePath": out_path.replace("json", "jpg"),
                        "imageData": "null",
                        "imageHeight": mask.shape[0], 
                        "imageWidth": mask.shape[1]}
    for group_id, miny, minx, maxy, maxx in boxes:
        lableme_template['shapes'].append({"label": class_mapping[group_id],
                                           "points": [
                                                [minx, miny],
                                                [maxx, maxy]
                                           ],
                                           "group_id": group_id,
                                           "shape_type": "rectangle",
                                           "flags": {}})

    with open(out_path, 'w') as f:
        json.dump(labelme_template, f, indent=4)

def display_sample(display_list):
    plt.figure(figsize=(18, 18))
    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def post_process_predictions(inputs_masks_and_names, class_mapping, apply_cca=False, visualize=False, write_json=False):
    for img, mask, name in inputs_masks_and_names:
        if apply_cca:
            mask = cca(mask, class_mapping)

        if visualize:
            display_sample([img, mask])

        if write_json:
            write_labelme_json(mask, class_mapping, name.replace("jpg", "json"))

        cv2.imwrite(name.replace("jpg", "png"))

def generic_seg_inference(model, input_imgs, img_names, class_mapping, is_gscnn=False, apply_cca=False, visualize=False, write_json=False):
    inputs_masks_and_names = []
    for img, name in zip(input_imgs, img_names):
        y_pred = model(img[tf.newaxis,...], training=False)
        if is_gscnn:
            y_pred = y_pred[...,:-1]  # gscnn has seg and shape head

        # TODO: the numpy call slows things down, can we do everything with tf operations?
        mask = create_mask(y_pred)[0].numpy()
        inputs_masks_and_names.append((img, mask, name))

    post_process_predictions(inputs_masks_and_names, class_mapping, apply_cca=apply_cca, visualize=visualize, write_json=write_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-image", help="Single image to segment.")
    input_group.add_argument("--input-folder", help="Folder of images to segment.")

    parser.add_argument("--saved-model", help="Directory or h5 file with a saved model.")
    parser.add_argument("--model", help='One of "unet", "fast_fcn", "deeplabv3plus", or "gated_scnn".')
    parser.add_argument("--img-size", type=int, help="Size of images. Should match the size trained on.")
    parser.add_argument("--saved-pkl", help='The saved PKL file from the training step. It is used for the class mapping.')
    parser.add_argument("--apply-cca", default=False, action='store_true', help="Post process with conncected component analysis. Makes segmentations uniform, but might miss objects.")
    parser.add_argument("--visualize", default=False, action='store_true', help="If set, will open a matplotlib plot to see the segmentation visually.")
    parser.add_argument("--write-annotation", default=False, action='store_true', help="If set, will also write a json file with annotations in labelme format.")
    
    args = parser.parse_args()
    
    # Load model
    model = tf.keras.load_model(args.saved_model, compile=False)
    

    # Figure out our mode (single or multi)
    is_single = True if args.input_image else False
    
    # Prepare inputs
    if is_single:
        input_img, img_name = prepare_single_image(args.input_image, args.img_size)
        input_imgs = [input_img]
        img_names = [img_names]
    else:
        img_names = []
        input_imgs = []
        for img in os.path.listdir(args.input_folder):
            img_path = os.path.join(args.input_folder, img)
            input_image, img_name = prepare_single_image(img_path, args.img_size)

            img_names.append(img_name)
            input_imgs.append(input_image)
    
    # Perform prediction with specified options
    _, class_mapping = pickle.load(open(args.saved_pkl, 'rb'))

    generic_seg_inference(model, input_imgs, img_names, class_mapping, is_gscnn="gated" in args.model, 
                                                                       apply_cca=args.apply_cca, 
                                                                       visualize=args.visualize, 
                                                                       write_json=args.write_annotation)
