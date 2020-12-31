import datasets.dad_gscnn
import datasets.publaynet_gscnn

from datasets.dad import build_dad_dataset
from datsets.publaynet import build_publaynet_dataset

def get_dataset(dataset, model):
    if dataset == "dad":
        if model == "gated_scnn":
            return dad_gscnn.build_dataset
        else:
            return build_dad_dataset
    elif dataset == "publaynet":
        if model == "gated_scnn":
            return publaynet_gscnn.build_dataset
        else:
            return build_publaynet_dataset
    else:
        raise NotImplementedError("Unsupported dataset {}.".format(dataset))

