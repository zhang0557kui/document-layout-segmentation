import argparse
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def prepare_single_image(input_img):
    raise NotImplementedError("I DONT FEEL LIKE IT!")

def prepare_image_folder(folder_path, batch_size):
    raise NotImplementedError("NO!")

def cca(seg_mask):
    raise NotImplementedError("NOT IN YOUR LIFE!")

def gated_scnn_seg_inference(model, input_img):
    raise NotImplementedError("NOPE!")

def generic_seg_inference(model, input_img):
    raise NotImplementedError("NEVER!")

if __name__ == '__main__':
    parser = argparse.parser()
    parser.add_argument("--saved-model", help="Directory or h5 file with a saved model.")
    parser.add_argument("--model", help'One of "unet", "fast_fcn", "deeplabv3plus", or "gated_scnn".')
    parser.add_argument("--img-size", type=int, help="Size of images. Should match the size trained on.")
    parser.add_argument("--saved-pkl", help='The saved PKL file from the training step. It is used for the class mapping.')
    parser.add_argument("--input-image", default="", help="Single image to segment.")
    parser.add_argument("--input-folder", default="", help="Folder of images to segment.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size of predictions. Default 1.")
    parser.add_argument("--apply-cca", type=bool, default=False, action='store_true', help="Post process with conncected component analysis. Makes segmentations uniform, but might miss objects.")

    args = parser.parse_args()

    model = tf.keras.load_model(args.saved_model, compile=False)
    
