
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=3)

model.save("digit_model.h5")
print("Model saved as digit_model.h5")

loaded_model = load_model("digit_model.h5")
print("Model loaded successfully")

predictions = loaded_model.predict(x_test[:5])

print("\nPredictions:")
for i in range(5):
    predicted_digit = np.argmax(predictions[i])
    print(f"Image {i+1}: Predicted = {predicted_digit}, Actual = {y_test[i]}")