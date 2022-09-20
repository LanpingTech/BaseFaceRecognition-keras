import numpy as np
from PIL import Image
from keras.optimizers import Adam
import math
from functools import partial

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def preprocess_input(image):
    image /= 255.0 
    return image

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split("\t")
        labels.append(int(path_split[1]))
    num_classes = np.max(labels) + 1
    return num_classes

def get_lr_scheduler(lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr


    warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
    warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
    no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
    func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    return func

def get_optimizer(batch_size, lr, epochs):
    nbs = 64
    lr_max = 1e-3
    lr_min = 3e-4
    lr_init = min(max(batch_size / nbs * lr, lr_min), lr_max)
    lr_schr_min = min(max(batch_size / nbs * lr * 1e-2, lr_min * 1e-2), lr_max * 1e-2)

    optimizer = Adam(lr=lr_init, beta_1=0.9)
    lr_scheduler_func = get_lr_scheduler(lr_init, lr_schr_min, epochs)

    return optimizer, lr_scheduler_func

def read_annotation(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    np.random.seed(2022)
    np.random.shuffle(lines)
    num_val = int(len(lines) * 0.01)
    num_train = len(lines) - num_val
    return lines[:num_train], lines[num_train:]
