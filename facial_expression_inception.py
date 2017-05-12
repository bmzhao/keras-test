import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split



def preprocess(img):
    colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(colored, (299, 299))


def loadInputData(filename):
    input = np.loadtxt(filename, np.uint8, delimiter=',')
    input = input.reshape(input.shape[0], img_rows, img_cols, 1)
    return np.array(list(map(preprocess, input)))



def loadTargetData(filename, num_classes):
    target = np.loadtxt(filename, np.uint8)
    return keras.utils.to_categorical(target, num_classes)


if __name__ == '__main__':

    batch_size = 100
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

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validation, y_validation))

    print('Saving Model...')
    model.save(os.path.join(resources_dir, 'modelStage1.h5'))

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    model.save(os.path.join(resources_dir, 'model.h5'))

    print(np.argmax(model.predict(x_test), axis=1))
