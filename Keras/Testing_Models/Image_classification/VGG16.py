from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.preprocessing import image
import numpy as np
import keras
import cv2 as cv

# Using VGG16 to extract features:

model = VGG16(weights='imagenet')
img_path = '../../../Data/in/snail-shell.jpg'
# using opencv instead of Pillow
img = cv.imread(img_path)
blank_image = np.zeros((224, 224, 3), np.uint8)
img = cv.resize(img, (224, 224))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

label = decode_predictions(features)
print(label)
