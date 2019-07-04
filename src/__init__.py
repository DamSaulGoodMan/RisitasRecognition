from time import time

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1. / 255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
validation_data = ImageDataGenerator(rescale=1. / 255)

neurones_model = Sequential()


def load_from_dir(generator_data_type, dir_name):
	return generator_data_type.flow_from_directory(dir_name, target_size=(136, 102), batch_size=64, class_mode="binary")


def add_neurone_layer(list_layer_spec):
	for layer_spec in list_layer_spec:
		neurones_model.add(Dense(activation=layer_spec[0], units=layer_spec[1]))


neurones_model.add(Conv2D(32, (3, 3), input_shape=(136, 102, 3), activation="relu"))
neurones_model.add(MaxPooling2D(pool_size=(2, 2)))
neurones_model.add(Flatten())

add_neurone_layer([["sigmoid", 64], ["sigmoid", 128], ["sigmoid", 128], ["sigmoid", 1]])

sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)

# logs with otimizer="rmsprop" : 392986
neurones_model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy"])

trains = load_from_dir(train_data, "data/train")
validations = load_from_dir(train_data, "data/validation")

# To run tensorboard web app $> tensorboard --logdir=logs/
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

neurones_model.fit_generator(trains,
                             steps_per_epoch=400,
                             epochs=100,
                             validation_data=validations,
                             validation_steps=75,
                             callbacks=[tensorboard])

neurones_model.save_weights('model/save_' + str(time()) + '.h5')
