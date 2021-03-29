from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten


def encode_labels(labels):
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels


def split(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=100)
    return X_train, X_val, y_train, y_val


def cnn(images, labels):
    num_classes = labels.shape[-1]
    height = images.shape[-2]
    width = images.shape[-1]

    # Creating a Sequential model
    model = tf.keras.Sequential([
        Flatten(input_shape=(width, height)),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    return model


def training(model, X_train, y_train, X_val, y_val):
    filepath = 'model/model-cnn.h5'

    es_callback = EarlyStopping(monitor='val_loss', patience=4)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=50,
                        callbacks=[checkpoint, es_callback])

    load_model = models.load_model(filepath)
    acc, loss = load_model.evaluate(X_val, y_val)
    return history, acc, loss
