from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

train_valid_data = ImageDataGenerator()
test_valid_data = ImageDataGenerator()

train_unavailable_data = ImageDataGenerator()
test_unavailable_data = ImageDataGenerator()


def load_from_dir(generator_data_type, dir_name):
	return generator_data_type.flow_from_directory(dir_name, target_size=(136, 102), batch_size=1, class_mode="binary")


neurones_model = Sequential()

neurones_model.add(Conv2D(32, (3, 3), input_shape=(136, 102, 3), activation="relu"))
neurones_model.add(MaxPooling2D(pool_size=(2, 2)))
neurones_model.add(Flatten())

neurones_model.add(Dense(activation="relu", units=128))
neurones_model.add(Dense(activation="sigmoid", units=1))

neurones_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

trains = load_from_dir(train_valid_data, "DataSet/Risitas/Training")
print(trains.filenames)

neurones_model.fit_generator(load_from_dir(train_valid_data, "DataSet/Risitas/Training"),
                             steps_per_epoch=100,
                             epochs=10,
                             validation_data=load_from_dir(test_valid_data, "DataSet/Risitas/Test"),
                             validation_steps=100)
