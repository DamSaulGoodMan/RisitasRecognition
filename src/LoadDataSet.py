import os


def load_from_dir(dir_name, is_from_valid_data_set):
	return {[is_from_valid_data_set, file] for file in os.listdir(dir_name)}


def get_data_test(dir_valid_data_set, dir_invalid_data_test):
	return load_from_dir(dir_valid_data_set, True).union(load_from_dir(dir_invalid_data_test, False))

