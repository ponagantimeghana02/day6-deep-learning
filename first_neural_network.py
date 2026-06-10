from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

iris = load_iris()

X = iris.data
y = iris.target

y = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(4,)))

model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

predictions = model.predict(X_test)

predicted_classes = predictions.argmax(axis=1)
actual_classes = y_test.argmax(axis=1)

accuracy = accuracy_score(actual_classes, predicted_classes)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")