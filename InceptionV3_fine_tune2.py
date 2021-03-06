from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import img_proc_scikit
import tensorflow as tf
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


if __name__ == '__main__':
    train_df = img_proc_scikit.TrainData('train.csv').make_train()
    class_weights = img_proc_scikit.TrainData('train.csv').calc_weights(labels_dict=img_proc_scikit.TrainData('train.csv').make_histdata(train_df), method='simple')

    img_rows, img_cols = 299, 299  # Resolution of inputs
    channel = 3
    num_classes = 28
    batch_size = 64
    epochs = 20

    train_data_dir = 'classes_dir'
    #    validation_data_dir = 'data/validation'
    nb_train_samples = 50782
    #    nb_validation_samples = 800

    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')


def InceptionV3_make(weights_file, classes_num):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(classes_num, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    """
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy', f1])
    
    # Start Fine-tuning
    model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, verbose=1, class_weight=class_weights)
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.
    
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)
    """
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', f1])
    model.load_weights(weights_file)
    return model
epochs = 50
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
if __name__ == '__main__':
    model = InceptionV3_make('InceptionV3_ft.h5', 28)
    model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, verbose=1, class_weight=class_weights)
    model.save_weights('InceptionV3_ft2.h5')
