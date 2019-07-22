from time import time

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2


def init_data():
    train_data = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
    validation_data = ImageDataGenerator(rescale=1. / 255)

    return train_data, validation_data


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


def run_training(optimizer_used, name_for_export):
    neurones_model.compile(optimizer=optimizer_used, loss="binary_crossentropy", metrics=["accuracy"])

    trains = load_from_dir(data[0], "data/train")
    validations = load_from_dir(data[1], "data/validation")

    # To run tensorboard web app $> tensorboard --logdir=logs/
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()) + name_for_export)

    neurones_model.fit_generator(trains,
                                 steps_per_epoch=150,
                                 epochs=40,
                                 validation_data=validations,
                                 validation_steps=100,
                                 callbacks=[tensorboard])

    neurones_model.save('model/save_' + name_for_export + str(time()) + '.hdf5')


# Firs Model with one neurone
# def shape_linear_model():
#     neurones_model.add(Conv2D(16, (1, 1), strides=(2, 2),
#                               input_shape=(136, 102, 3),
#                               kernel_initializer=kernel_init,
#                               kernel_regularizer=l2(kernel_reg), activation="relu"))
#     neurones_model.add(Flatten())
#
#     add_neurone_layer([["sigmoid", 1]])
#
#
# neurones_model = Sequential()
# data = init_data()
# shape_linear_model()
# run_training("adam", "linear_model")


def shape_last_model():
    neurones_model.add(Conv2D(16, (3, 3), strides=(2, 2),
                              input_shape=(136, 102, 3),
                              kernel_initializer=kernel_init,
                              kernel_regularizer=l2(kernel_reg), activation="relu"))

    con2d_add(32)
    con2d_add(64)
    con2d_add(128)
    neurones_model.add(Flatten())

    add_neurone_layer([["sigmoid", 128]])
    neurones_model.add(BatchNormalization())
    neurones_model.add(Dropout(0.5))

    add_neurone_layer([["sigmoid", 1]])


neurones_model = Sequential()
data = init_data()
shape_last_model()
run_training("adam", "")


def shape_last_2_model():
    neurones_model.add(Conv2D(16, (3, 3), strides=(2, 2),
                              input_shape=(136, 102, 3),
                              kernel_initializer=kernel_init,
                              kernel_regularizer=l2(kernel_reg), activation="relu"))


    neurones_model.add(Flatten())

    add_neurone_layer([["sigmoid", 32]])
    add_neurone_layer([["sigmoid", 32]])
    neurones_model.add(BatchNormalization())
    neurones_model.add(Dropout(0.5))

    add_neurone_layer([["sigmoid", 1]])


neurones_model = Sequential()
data = init_data()
shape_last_2_model()
run_training("adam", "deepLayer_32_1_conv2d_model")


def shape_model():
    neurones_model.add(Conv2D(16, (3, 3), strides=(2, 2),
                              input_shape=(136, 102, 3),
                              kernel_initializer=kernel_init,
                              kernel_regularizer=l2(kernel_reg), activation="relu"))
    con2d_add(32)
    con2d_add(64)

    neurones_model.add(Flatten())

    add_neurone_layer([["sigmoid", 16]])
    neurones_model.add(BatchNormalization())
    neurones_model.add(Dropout(0.5))

    add_neurone_layer([["sigmoid", 1]])


neurones_model = Sequential()
data = init_data()
shape_model()
run_training("adam", "deepLayer_16_1_conv2d_model")


def shape_model():
    neurones_model.add(Conv2D(16, (2, 2), strides=(2, 2),
                              input_shape=(136, 102, 3),
                              kernel_initializer=kernel_init,
                              kernel_regularizer=l2(kernel_reg), activation="relu"))

    con2d_add(32)
    con2d_add(64)
    # con2d_add(128)
    neurones_model.add(Flatten())

    add_neurone_layer([["sigmoid", 8]])
    add_neurone_layer([["sigmoid", 16]])
    add_neurone_layer([["sigmoid", 32]])
    neurones_model.add(BatchNormalization())
    neurones_model.add(Dropout(0.5))

    add_neurone_layer([["sigmoid", 1]])


neurones_model = Sequential()
data = init_data()
shape_model()
run_training("adam", "3deepLayer_4_8_16_3conv2d_model")


# sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
