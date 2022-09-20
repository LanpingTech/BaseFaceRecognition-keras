from PIL import Image
import numpy as np

from model import facenet
from utils import preprocess_input, resize_image

def detect_image(model, image_1, image_2):
    image_1 = resize_image(image_1, [160, 160], True)
    image_2 = resize_image(image_2, [160, 160], True)
    
    photo_1 = np.expand_dims(preprocess_input(np.array(image_1, np.float32)), 0)
    photo_2 = np.expand_dims(preprocess_input(np.array(image_2, np.float32)), 0)

    output1 = model.predict(photo_1)
    output2 = model.predict(photo_2)

    l1 = np.linalg.norm(output1-output2, axis=1)
    return l1

if __name__ == "__main__":
    model = facenet((160, 160, 3), mode='predict')
    model.load_weights('runs/facenet.h5')
        
    while True:
        image_1 = input('Input image_1 filename:')
        image_1 = Image.open(image_1)


        image_2 = input('Input image_2 filename:')
        image_2 = Image.open(image_2)
        
        probability = detect_image(model, image_1, image_2)
        print(probability)