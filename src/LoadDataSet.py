from keras.preprocessing.image import ImageDataGenerator

train_valid_data = ImageDataGenerator()
test_valid_data = ImageDataGenerator()

train_unavailable_data = ImageDataGenerator()
test_unavailable_data = ImageDataGenerator()


def load_from_dir(generator_data_type, dir_name):
	return generator_data_type\
			.flow_from_directory(dir_name, target_size=(136, 102), batch_size=32, class_mode="binary")
