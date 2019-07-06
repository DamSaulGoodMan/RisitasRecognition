from time import time

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2


train_data = ImageDataGenerator(rescale=1. / 255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
validation_data = ImageDataGenerator(rescale=1. / 255)

neurones_model = Sequential()


def load_from_dir(generator_data_type, dir_name):
	return generator_data_type.flow_from_directory(dir_name, target_size=(136, 102), batch_size=64, class_mode="binary")

kernel_init = "glorot_uniform"
kernel_reg = 0.0001
axis_value = 1


def add_neurone_layer(list_layer_spec):
	for layer_spec in list_layer_spec:
		neurones_model.add(Dense(activation=layer_spec[0], units=layer_spec[1], kernel_initializer=kernel_init))


def con2d_add(kernel_size):
	neurones_model.add(Conv2D(kernel_size, (2, 2),
	                          padding="same",
	                          kernel_initializer=kernel_init,
	                          kernel_regularizer=l2(kernel_reg),
	                          activation="relu"))

	neurones_model.add(BatchNormalization(axis=axis_value))

	neurones_model.add(Conv2D(kernel_size, (2, 2),
	                          padding="same",
	                          kernel_initializer=kernel_init,
	                          kernel_regularizer=l2(kernel_reg),
	                          strides=(2, 2),
	                          activation="relu"))
	neurones_model.add(BatchNormalization(axis=axis_value))
	neurones_model.add(Dropout(0.25))

neurones_model.add(Conv2D(16, (3, 3), strides=(2, 2),
                          input_shape=(136, 102, 3),
                          kernel_initializer=kernel_init,
                          kernel_regularizer=l2(kernel_reg), activation="relu"))

con2d_add(32)
con2d_add(64)
con2d_add(128)
neurones_model.add(Flatten())

add_neurone_layer([["sigmoid", 256]])
neurones_model.add(BatchNormalization())
neurones_model.add(Dropout(0.5))

add_neurone_layer([["sigmoid", 1]])


sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)

# logs with otimizer="rmsprop" : 392986
neurones_model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy"])

trains = load_from_dir(train_data, "data/train")
validations = load_from_dir(train_data, "data/validation")

# To run tensorboard web app $> tensorboard --logdir=logs/
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

neurones_model.fit_generator(trains,
                             steps_per_epoch=200,
                             epochs=150,
                             validation_data=validations,
                             validation_steps=100,
                             callbacks=[tensorboard])

neurones_model.save_weights('model/save_' + str(time()) + '.h5')
