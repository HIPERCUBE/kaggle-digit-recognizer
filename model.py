from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout


def simple_model():
    """
    Score: 0.97xx
    """
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def simple_cnn():
    """
    Score:
    """
    model = Sequential([

    ])
    model.compile()


def build_model():
    model = simple_model()
    model.summary()
    return model
