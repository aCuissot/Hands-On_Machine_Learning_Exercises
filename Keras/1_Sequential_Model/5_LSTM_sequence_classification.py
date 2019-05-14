from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras

max_features = 1024

x_train = np.random.random(max_features)
y_train = keras.utils.to_categorical(np.random.randint(1, size=max_features))
x_test = np.random.random(100)
y_test = keras.utils.to_categorical(np.random.randint(1, size=100))

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
print(score)
