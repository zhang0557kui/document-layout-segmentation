import argparse

def train(model, train, valid, test):
    raise NotImplementedError("No Training for you!")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation model.')
    parser.add_argument('--model', help='One of "unet", "fast_fcn", "gated_scnn", or "deeplabv3plus".')
    parser.add_argument('--num-classes', type=int, help='Number of classes to segment, including background.')
    parser.add_argument('--ignore-class', type=int, default=255, help='Class number to ignore. Defualt 255.')
    parser.add_argument('--dataset', help='Either "dad" or "publaynet".')
    parser.add_argument('--dataset-dir', help='Root folder of the dataset.')
    parser.add_argument('--pre-split', defualt=None, help='If set, assume there is a train, test, and val dir in the dataset dir.')
    
    # Parse the command args
    args = parser.parse_args()
    
    # Build the requested dataset and get the int->label class mapping
    print("Building dataset...\n")
    train, valid, test, class_mapping = build_dataset(args.dataset, args.dataset_dir, args.pre_split, args.ignore_class)

    # Build the specified segmentation model
    print("Building model...\n")
    model = build_model(args.model, args.num_classes)

    # Train the model
    print("Starting train loop...\n")
    train(model, train, valid)
    
    # Report stats from the test set
    print("Gather accuracy statistics...\n")
    report_results(model, test, class_mapping)

    print("\nCOMPLETE\n")

