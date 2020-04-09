from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Flatten, Conv2D,MaxPooling2D

model = Sequential([
    Input(shape=(100,100,3)),
    Conv2D(16,3,padding="same",activation="relu"),
    MaxPooling2D(),
    Conv2D(16,3,padding="same",activation="relu"),
    MaxPooling2D(),
    Conv2D(8,3,padding="same",activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
model.save("./model/model.h5")