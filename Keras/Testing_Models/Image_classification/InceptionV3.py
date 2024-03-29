from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from keras.preprocessing import image
import numpy as np

model = InceptionV3(weights='imagenet')

img_path = '../../../Data/in/snail-shell.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
