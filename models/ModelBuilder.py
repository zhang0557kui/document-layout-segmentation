from unet import build_unet
from gated_scnn import build_gscnn
from fast_fcn import build_fast_fcn
from deeplabv3plus import build_deeplabv3plus

MODEL_DICT = {"unet": build_unet,
              "gated_scnn": build_gscnn,
              "fast_fcn": build_fast_fcn,
              "deeplabv3+" build_deeplabv3plus}

def build_model(name, img_size, num_channels):
    if name in MODEL_DICT:
        return MODEL_DICT[name].build(img_size, num_channels)
    else:
        raise NotImplementedError("{} is not a supported model".format(name))

