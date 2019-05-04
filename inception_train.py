import numpy as np
import sys, os
import keras

from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

# K.set_learning_phase(1)

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"




def prepro_fn(x):
    return (x / 255. - 0.5) * 2  # from [0, 255] to [-1,+1]


def main_simd(src1, src2, test_only, model_path, model_name,
              load_weight, weight):
    batch_size = 32
    num_classes = 2
    epochs = 5
    x_train_num = 3000
    x_test_num = 1500
    image_size = 256

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='max')
    auto_save_callback = keras.callbacks.ModelCheckpoint(os.path.join(model_path, model_name),
                                                         monitor='val_acc',
                                                         mode='max', save_best_only=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

    print('Using real-time data augmentation.')
    train_datagen = ImageDataGenerator(preprocessing_function=prepro_fn,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       # rescale=1. / 255,
                                       vertical_flip=False,
                                       horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        src1,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    test_datagen = ImageDataGenerator(preprocessing_function=prepro_fn)

    validation_generator = test_datagen.flow_from_directory(
        src2,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    # Get the model and compile it.
    # img_input = keras.layers.Input(shape=(image_size, image_size, 3))

    # base_model = MobileNet
    base_model = InceptionV3

    base_model = base_model(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if load_weight is True:
        model.load_weights(weight)

    model.summary()


    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if test_only is not True:
        print("Training model...")
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=x_train_num // batch_size,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=x_test_num // batch_size,
                            callbacks=[auto_save_callback, earlystopping, tensorboard]
                            )
    else:
        print("Testing...")

    # Evaluate model with test data set and share sample prediction results
    evaluation = model.evaluate_generator(validation_generator,
                                          steps=x_test_num // batch_size)


    print('Model Accuracy = %.4f' % (evaluation[1]))


def main():
    src1 = './train_all/'
    src2 = './validation_all/'


    save_model_path = './inception_models/'
    model_name = 'gan_all_model.h5'  # for saving model

    # tags: True to enter testing phase.
    test_only = True

    # if load_weight == True, our model will be initialized by our weights.
    load_weight = False

    # default: load our weights from the previous model.(namely model_car, model_person,...)
    weight = './inception_models/gan_good_model.h5'

    main_simd(src1, src2, test_only, save_model_path, model_name,
              load_weight, weight)


if __name__ == '__main__':
    main()
