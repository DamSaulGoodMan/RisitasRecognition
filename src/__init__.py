from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

neurones_model = Sequential()


neurones_model.add(Convolution2D(64, 3, 3, input_shape=(136, 102, 3), activation="relu"))
neurones_model.add(MaxPooling2D(pool_size=(2, 2)))
neurones_model.add(Flatten())

neurones_model.add(Dense(output_dim=128, activation="relu"))
neurones_model.add(Dense(output_dim=1, activation="sigmoid"))

