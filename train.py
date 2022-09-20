import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import datetime

from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard

from model import facenet, triplet_loss
from dataset import WebFacesDataset
from utils import num_classes, get_optimizer, read_annotation

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotation_path', default='annotation.txt')

    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='runs')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=1e-3)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    n_classes = num_classes(args.annotation_path)

    input_shape = (160, 160, 3)
    model = facenet(input_shape, n_classes)
    if args.pretrained != '':
        model.load_weights(args.pretrained, by_name=True, skip_mismatch=True)

    optimizer, lr_scheduler_func = get_optimizer(args.batch_size, args.lr, args.epochs)
    model.compile(
        loss={'Embedding' : triplet_loss(batch_size=args.batch_size//3), 'Softmax' : 'categorical_crossentropy'}, 
        optimizer = optimizer, metrics = {'Softmax' : 'categorical_accuracy'}
    )
    lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose = 1)

    trainset, valset = read_annotation(args.annotation_path)
    train_dataset = WebFacesDataset(input_shape, trainset, args.batch_size, n_classes, random=True)
    val_dataset = WebFacesDataset(input_shape, valset, args.batch_size, n_classes, random=False)

    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(args.save_dir, str(time_str))
    logger = TensorBoard(log_dir)
    checkpoint = ModelCheckpoint(os.path.join(log_dir, "epoch{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
    callbacks = [
        # lr_scheduler,
        logger,
        checkpoint,
        early_stopping
    ]

    model.fit_generator(
        generator = train_dataset,
        steps_per_epoch = len(trainset) // args.batch_size,
        validation_data = val_dataset,
        validation_steps = len(valset) // args.batch_size,
        epochs= args.epochs,
        initial_epoch = 0,
        use_multiprocessing = True,
        workers = 4,
        callbacks = callbacks
    )

    # nohup python -u train.py > train.log 2>&1 &














