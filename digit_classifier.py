
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training Images Shape:", X_train.shape)
print("Testing Images Shape:", X_test.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = Sequential([
    
    # Convert 28x28 image into 784 values
    Flatten(input_shape=(28, 28)),
    
    # Hidden Layer 1
    Dense(128, activation='relu'),
    
    # Hidden Layer 2
    Dense(64, activation='relu'),
    
    # Output Layer
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train_cat,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

loss, accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

print("\nSample Predictions")
print("-" * 50)

predictions = model.predict(X_test[:5], verbose=0)

for i in range(5):

    predicted_digit = np.argmax(predictions[i])

    confidence = np.max(predictions[i]) * 100

    actual_digit = y_test[i]

    print(f"\nImage {i+1}")
    print("Actual Digit    :", actual_digit)
    print("Predicted Digit :", predicted_digit)
    print("Confidence Score:", round(confidence, 2), "%")