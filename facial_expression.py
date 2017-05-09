import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    batch_size = 128
    epochs = 12
    num_classes = 3
    img_rows, img_cols = 48, 48

    resources_dir = os.path.join(os.path.dirname(__file__), 'resources')

    TRAIN_DATA_FILE = os.path.join(resources_dir, 'train_data_limited.csv')
    TRAIN_TARGET_FILE = os.path.join(resources_dir,'train_target_limited.csv')
    TEST_DATA_FILE = os.path.join(resources_dir,'test_data.csv')


    x_train = np.loadtxt(TRAIN_DATA_FILE, np.int32, delimiter=',')
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    y_train = np.loadtxt(TRAIN_TARGET_FILE, np.int8)
    y_train = keras.utils.to_categorical(y_train, num_classes)






    # #vggnet model
    # model = Sequential([
    #     Conv2D(64, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 1), activation='relu', padding='same'),
    #     Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'),
    #     Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     Dense(num_classes, activation='softmax')
    # ])
    #
    # model.compile(optimizer=keras.optimizers.Adam(),
    #               loss=keras.losses.categorical_crossentropy,
    #               metrics=['accuracy'])
    #
    #
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    #
    #
    #
    # # score = model.evaluate(x_test, y_test, verbose=0)
    # #
    # # print('Test loss:', score[0])
    # # print('Test accuracy:', score[1])






