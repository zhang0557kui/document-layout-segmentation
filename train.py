import argparse

from loss import seg_loss, SegmentationAccuracy
from models.ModelBuilder import build_model

def report_results(model, test, valid, class_mapping):
    raise NotImplementedError("No results for you!")

def train(model, train, valid, test, lr, patience, model_name):
    def calc_loss(model, input_image, gt_mask, gt_boxes, training):
        predicted_mask = model(input_image, training=training)
        return seg_loss(gt_mask, predicted_mask, gt_boxes)

    def grad(model, input_image, gt_mask, gt_boxes):
        with tf.GradientTape() as tape:
            loss_value = calc_loss(model, input_image, gt_mask, gt_boxes, True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    best_val_loss = 1000000.0
    num_bad_iters = 0
    num_epochs = 100
    lr_decreased = False
    for epoch in range(num_epochs):
        if num_bad_iters >= patience and lr_decreased:
            print("Val Loss is not improving, exiting...")
            break
        elif num_bad_iters == patience:
            print("Lowering lr, restarting from best model")
            lr_decreased = True
            num_bad_iters = 0
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
            model = tf.keras.models.load_model(model_name, compile=False)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = SegmentationAccuracy()
    
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        epoch_val_accuracy = SegmentationAccuracy()
    
        step = 0
        for input_image, gt_mask, gt_boxes in train:
            loss_value, grads = grad(model, input_image, gt_mask, gt_boxes)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(gt_mask, model(input_image, training=True))

            step += 1
            if step % 100 == 0:
                print("Step {}: Loss: {:.3f}, Accuracy: {:.3%}".format(step, epoch_loss_avg.result(), epoch_accuracy.result()))
    
        for input_image, gt_mask, gt_boxes in valid:
            loss_value = loss(model, input_image, gt_mask, gt_boxes, training=False)
            epoch_val_loss_avg.update_state(loss_value)
            epoch_val_accuracy.update_state(gt_mask, model(input_image, training=False))
    
        val_loss = epoch_val_loss_avg.result()
        if val_loss < best_val_loss:
            print("Val Loss decreased from {:.4f} to {:.4f}".format(best_val_loss, val_loss))
            best_val_loss = val_loss
            num_bad_iters = 0
            model.save(model_name)
        else:
            print("Val Loss did not decrease from {:.4f}".format(best_val_loss))
            num_bad_iters += 1
    
        print("Epoch: {:02d} Loss: {:.3f}, Accuracy: {:.3%}, Val Loss: {:.3f}, Val Accuracy: {:.3%}\n".format(epoch, 
                                                                                                        epoch_loss_avg.result(),
                                                                                                        epoch_accuracy.result(),
                                                                                                        epoch_val_loss_avg.result(),
                                                                                                        epoch_val_accuracy.result()))       



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation model.')
    
    parser.add_argument('--model', help='One of "unet", "fast_fcn", "gated_scnn", or "deeplabv3plus".')
    parser.add_argument('--num-classes', type=int, help='Number of classes to segment, including background.')
    parser.add_argument('--ignore-class', type=int, default=255, help='Class number to ignore. Defualt 255.')
    parser.add_argument('--patience', type=int, default=5, help='Set how many epochs to wait for val loss to increase.')
    parser.add_argument('--base-lr', type=float, default=1.0e-4, help='Set initial learning rate. After val loss stops increasing for number of epochs specified by --patience, the model reloads to the best point and divides the learning rate by 10 for fine tuning')
    parser.add_argument('--use-box-loss', default=None, help='If set, use box loss regression during loss calculation')

    parser.add_argument('--dataset', help='Either "dad" or "publaynet".')
    parser.add_argument('--dataset-dir', help='Root folder of the dataset.')
    
    # Parse the command args
    args = parser.parse_args()

    # Build the requested dataset and get the int->label class mapping
    print("Building dataset...\n")
    if args.dataset == "dad":
        train, valid, test, class_mapping = build_dad_dataset(args.dataset_dir)
    elif args.dataset == "publaynet":
        train, valid, test, class_mapping = build_publaynet_dataset(args.dataset_dir)
    else:
        raise NotImplementedError("Unsupported dataset {}.".format(args.dataset))

    # Build the specified segmentation model
    print("Building model...\n")
    model = build_model(args.model, args.num_classes)

    # Train the model
    print("Starting train loop...\n")
    train(model, train, valid, args.base_lr, args.patience, args.model + "best.h5")
    
    # Report stats from the test set
    print("Gather accuracy statistics...\n")
    report_results(model, test, class_mapping)

    print("\nCOMPLETE\n")

