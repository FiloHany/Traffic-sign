from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

class TrafficSignModel:
    """Builds CNN model for traffic sign classification."""

    @staticmethod
    def build(input_shape, num_classes=43):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.15))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.20))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model