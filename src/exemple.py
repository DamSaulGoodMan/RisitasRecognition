from time import time

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# C'est des paramètre pour les couches Con2D, regarde sur internet pour la doc
kernel_init = "glorot_uniform"
kernel_reg = 0.0001
axis_value = 1

# Tu crée ton modèle de neurones, c'est un modèle simple ou les connections sont géré automatiquement entre neurones
neurones_model = Sequential()

# Tout les arguments ici c'est pour modifier les images quand tu les load et éviter que ton modèle vois toujours les
# meme images
train_data = ImageDataGenerator(rescale=1. / 255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
validation_data = ImageDataGenerator(rescale=1. / 255)

# Donc la on arrive sur les Con2D et Co. Les conv2d vont appliquer des algorithmes sur les images pour les rendre plus
# apte à etre reconnu par le modèle, plus simple (on essaye de décomplexifier les données pour avoir un modèle plus
# efficace sans ppour autant avoir 2000000 de couches de neurones !).
neurones_model.add(Conv2D(16, (3, 3), strides=(2, 2),
                              input_shape=(136, 102, 3),
                              kernel_initializer=kernel_init,
                              kernel_regularizer=l2(kernel_reg), activation="relu"))
neurones_model.add(Conv2D(64, (2, 2),
                          padding="same",
                          kernel_initializer=kernel_init,
                          kernel_regularizer=l2(kernel_reg),
                          activation="relu"))
neurones_model.add(BatchNormalization(axis=axis_value))
neurones_model.add(Conv2D(64, (2, 2),
                          padding="same",
                          kernel_initializer=kernel_init,
                          kernel_regularizer=l2(kernel_reg),
                          strides=(2, 2),
                          activation="relu"))
neurones_model.add(BatchNormalization(axis=axis_value))
neurones_model.add(Dropout(0.25))

neurones_model.add(Flatten())

# Couche de neurones qui contient les paramètre de ton modèle
neurones_model.add(Dense(activation="sigmoid", units=256, kernel_initializer=kernel_init))
neurones_model.add(BatchNormalization())
neurones_model.add(Dropout(0.5))

# Neurone unique de sortie qui en gros renvoi oui ou non, si mon image est Issou.
neurones_model.add(Dense(activation="sigmoid", units=1, kernel_initializer=kernel_init))

# tu compile ton modèle >> va voir sur internet c'est assez simple
neurones_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# La tu charge les data set, attention au  !!!! chemin !!!!
trains = train_data.flow_from_directory("data/train", target_size=(136, 102), batch_size=64, class_mode="binary")
validations = validation_data.flow_from_directory("data/validation", target_size=(136, 102), batch_size=64,
                                                  class_mode="binary")
# Donc la tu a 2 data set avec un d'entrainement et l'autre pour valider l'entrainement
# l'objectif c'est d'avoir environ les meme mesures sur les deux data set

# To run tensorboard web app $> tensorboard --logdir=logs/
# hésite pas à run le tensor board, c'est plus potable les graphique que les métric brut !
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# la tu entraine ton modèle, steps c'est conbien de fois tu lui passe une image et lui demande de quelle data set elle
# appartient
# epoch c'est le nombre de fois que tu répète l'action précédente ET la meme pour la validation apres
neurones_model.fit_generator(trains,
                             steps_per_epoch=150,
                             epochs=60,
                             validation_data=validations,
                             validation_steps=75,
                             callbacks=[tensorboard])

# le modèle obtenue que tu export pour le site web
neurones_model.save('model/save_' + str(time()) + '.hdf5')
