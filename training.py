import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models


def encode(X, y):
    X = np.array(X)
    y = to_categorical(y, dtype='uint8')
    return X, y


def split(X, y):
    X_train, y_train, X_val, y_val = train_test_split(X, y, random_state=100)
    return X_train, y_train, X_val, y_val


def cnn():
    num_classes = 6
    img_shape = 300, 300, 1

    model = Sequential()
    model.add(Conv2D(16, 5, activation='relu', input_shape=img_shape))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, 5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def training(model, X_train, y_train, X_val, y_val):
    filepath = 'model/model-cnn.h5'

    es_callback = EarlyStopping(monitor='val_loss', patience=4)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=50,
                        callbacks=[checkpoint, es_callback])

    model_dir = 'model/model-cnn.h5'

    load_model = models.load_model(model_dir)
    evaluate = load_model.evaluate(X_val, y_val)

    return history, evaluate
