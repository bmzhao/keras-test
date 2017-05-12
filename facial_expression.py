import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

def loadInputData(filename):
    input = np.loadtxt(filename, np.uint8, delimiter=',')
    input = input.reshape(input.shape[0], img_rows, img_cols, 1)
    return input


def loadTargetData(filename, num_classes):
    target = np.loadtxt(filename, np.uint8)
    return keras.utils.to_categorical(target, num_classes)



if __name__ == '__main__':

    batch_size = 300
    # batch_size = 10
    epochs = 100
    # epochs = 1

    num_classes = 3
    img_rows, img_cols = 48, 48

    resources_dir = os.path.join(os.path.dirname(__file__), 'resources')

    TRAIN_DATA_FILE = os.path.join(resources_dir, 'train_data.csv')
    TRAIN_TARGET_FILE = os.path.join(resources_dir, 'train_target.csv')
    TEST_DATA_FILE = os.path.join(resources_dir, 'test_data.csv')

    x_train = loadInputData(TRAIN_DATA_FILE)
    y_train = loadTargetData(TRAIN_TARGET_FILE, num_classes)

    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.10)

    x_test = loadInputData(TEST_DATA_FILE)


    #vggnet model
    # model = Sequential([
    #     Conv2D(64, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 1), activation='relu', padding='same'),
    #     Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #     Flatten(),
    #     Dense(4096, activation='relu'),
    #     Dense(4096, activation='relu'),
    #     Dense(num_classes, activation='softmax')
    # ])


    model = Sequential([
        Conv2D(64, kernel_size=(7, 7), input_shape=(img_rows, img_cols, 1), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, kernel_size=(7, 7), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(7, 7), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(7, 7), activation='relu', padding='same'),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1)
    rmsprop = optimizers.RMSprop(lr=0.00001)

    model.compile(optimizer=rmsprop,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])


    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validation, y_validation), callbacks=[earlyStop])

    print('Saving Model...')
    model.save(os.path.join(resources_dir, 'modelStage1.h5'))


    # score = model.evaluate(x_test, y_test, verbose=0)
    #
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])






