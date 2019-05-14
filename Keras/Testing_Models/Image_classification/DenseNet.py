from keras.applications.densenet import decode_predictions, DenseNet201, preprocess_input
from keras.preprocessing import image
import numpy as np

model = DenseNet201(weights='imagenet')

img_path = '../../../Data/in/snail-shell.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
