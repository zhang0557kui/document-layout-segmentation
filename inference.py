import argparse
import os
import statistics
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from skimage import measure


def prepare_single_image(input_img_path, img_size):
    # Load image data
    assert input_img_path.endswith(".jpg"), "Images must be jpg format"
    input_img = tf.io.read_file(img_path)
    input_img = tf.image.decode_jpeg(image, channels=3)

    # Resize and normalize
    input_img = tf.image.resize(input_img, (img_size, img_size))
    input_img = tf.math.divide(tf.cast(input_image, tf.float32), 255.0) 

    return input_img, os.path.basename(input_img_path)

def cca(pred_mask, out_path, class_mapping, min_area=100, write_boxes=False):
    new_mask = pred_mask.copy()
    lbl = measure.label(pred_mask)

    regions = measure.regionprops(lbl)

    if write_boxes:
        box_f = open(out_path.replace("png", "csv"), 'w')
        box_f.write("class_id,class_name,min_y,min_x,max_y,max_x\n")

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
        if region_label != 0:
            new_mask[minr:maxr, minc:maxc] = [region_label]

        if write_boxes:
            box_f.write("{},{},{},{},{},{}\n".format(region_label, 
                                                     class_mapping[region_label]
                                                     minr,
                                                     minc,
                                                     maxr,
                                                     maxc))
    
    if write_boxes:
        box_f.close()
    return new_mask

def write_labelme_json(extracted_boxes, out_path):
    raise NotImplementedError("UGH THERES MORE???")

def post_process_single(y_pred, y_true, img_name, apply_cca=False, visualize=False):
    raise NotImplementedError("OH BOY")

def post_process_multiple(y_pred, y_true, img_names, apply_cca=False, visualize=False):
    raise NotImplementedError("IM BUSY OK!")

def gated_scnn_seg_inference(model, input_imgs, img_names, apply_cca=False, visualize=False, write_json=False):
    raise NotImplementedError("NOPE!")

def generic_seg_inference(model, input_imgs, img_names, apply_cca=False, visualize=False, write_json=False):
    raise NotImplementedError("NEVER!")

if __name__ == '__main__':
    parser = argparse.parser()
    parser.add_argument("--saved-model", help="Directory or h5 file with a saved model.")
    parser.add_argument("--model", help'One of "unet", "fast_fcn", "deeplabv3plus", or "gated_scnn".')
    parser.add_argument("--img-size", type=int, help="Size of images. Should match the size trained on.")
    parser.add_argument("--saved-pkl", help='The saved PKL file from the training step. It is used for the class mapping.')
    parser.add_argument("--apply-cca", type=bool, default=False, action='store_true', help="Post process with conncected component analysis. Makes segmentations uniform, but might miss objects.")
    parser.add_argument("--visualize", type=bool, defualt=False, action='store_true', help="If set, will open a matplotlib plot to see the segmentation visually.")
    parser.add_argument("--write-annotation", type=bool, defualt=False, action='store_true', help="If set, will also write a json file with annotations in labelme format.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-image", help="Single image to segment.")
    input_group.add_argument("--input-folder", help="Folder of images to segment.")
    
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

    if "gated" in args.model:
        gated_scnn_seg_inference(model, input_imgs, img_names, apply_cca=args.apply_cca, visualize=args.visualize, write_json=args.write_annotation)
    else:
        generic_seg_inference(model, input_imgs, img_names, apply_cca=args.apply_cca, visualize=args.visualize, w    rite_json=args.write_annotation)
