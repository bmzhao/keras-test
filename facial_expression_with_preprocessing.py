import dlib
import numpy as np
import keras
import os
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from sklearn.model_selection import train_test_split


predictor = dlib.shape_predictor('/home/bmzhao/Code/keras-test/resources/shape_predictor_68_face_landmarks.dat')


def preprocess(img):
    shape = predictor(img, dlib.rectangle(0, 0, 48, 48))
    return np.array([[shape.part(idx).x, shape.part(idx).y] for idx in range(68)]).flatten()


def loadInputData(filename):
    input = np.loadtxt(filename, np.uint8, delimiter=',')
    input = input.reshape(input.shape[0], img_rows, img_cols)
    return np.array(list(map(preprocess, input)))


def loadTargetData(filename, num_classes):
    target = np.loadtxt(filename, np.uint8)
    return keras.utils.to_categorical(target, num_classes)


# https://stackoverflow.com/questions/43011070/keras-concatenating-model-flattened-output-with-vector
if __name__ == '__main__':

    batch_size = 100
    # batch_size = 10
    epochs = 100
    # epochs = 1

    num_classes = 3
    img_rows, img_cols = 48, 48


    resources_dir = os.path.join(os.path.dirname(__file__), 'resources')

    TRAIN_DATA_FILE = os.path.join(resources_dir, 'train_data.csv')
    PREPROCESSED_TRAIN_DATA = os.path.join(resources_dir, 'train_data_preprocessed.csv')
    TRAIN_TARGET_FILE = os.path.join(resources_dir, 'train_target.csv')
    TEST_DATA_FILE = os.path.join(resources_dir, 'test_data.csv')

    # x_train = loadInputData(TRAIN_DATA_FILE).astype(np.uint8)
    # np.savetxt(PREPROCESSED_TRAIN_DATA, x_train)

    x_train = np.loadtxt(PREPROCESSED_TRAIN_DATA, np.uint8)

    y_train = loadTargetData(TRAIN_TARGET_FILE, num_classes)

    y_train.histogram()

    # x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.10)
    #
    # x_test = loadInputData(TEST_DATA_FILE)
    #
    # print('finished loading input data')
    #
    # model = Sequential([
    #     Dense(500, activation='relu', input_dim=136),
    #     Dense(500, activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(num_classes, activation='softmax')
    # ])
    #
    # earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1)
    # rmsprop = optimizers.RMSprop(lr=0.00001)
    #
    # model.compile(optimizer=rmsprop,
    #               loss=keras.losses.categorical_crossentropy,
    #               metrics=['accuracy'])
    #
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
    #           validation_data=(x_validation, y_validation))
    #
    # print('Saving Model...')
    # model.save(os.path.join(resources_dir, 'modelUsingPreprocessing.h5'))