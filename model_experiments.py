# model_experiments.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# One-hot encoding
y = to_categorical(y, 3)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Normalize Features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Different Model Configurations
experiments = [
    {
        "name": "Model 1",
        "layers": [8],
        "activation": "relu",
        "epochs": 30
    },
    {
        "name": "Model 2",
        "layers": [16],
        "activation": "relu",
        "epochs": 50
    },
    {
        "name": "Model 3",
        "layers": [16, 8],
        "activation": "relu",
        "epochs": 50
    },
    {
        "name": "Model 4",
        "layers": [32, 16],
        "activation": "tanh",
        "epochs": 100
    }
]

print("\nMODEL COMPARISON")
print("-" * 60)

results = []

for exp in experiments:

    model = Sequential()

    # First Hidden Layer
    model.add(Dense(
        exp["layers"][0],
        activation=exp["activation"],
        input_shape=(4,)
    ))

    # Additional Hidden Layers
    for neurons in exp["layers"][1:]:
        model.add(Dense(
            neurons,
            activation=exp["activation"]
        ))

    # Output Layer
    model.add(Dense(3, activation="softmax"))

    # Compile
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    model.fit(
        X_train,
        y_train,
        epochs=exp["epochs"],
        batch_size=8,
        verbose=0
    )

    # Predict
    predictions = model.predict(X_test, verbose=0)

    predicted_classes = predictions.argmax(axis=1)
    actual_classes = y_test.argmax(axis=1)

    accuracy = accuracy_score(
        actual_classes,
        predicted_classes
    )

    results.append(
        (
            exp["name"],
            len(exp["layers"]),
            exp["layers"],
            exp["activation"],
            exp["epochs"],
            round(accuracy * 100, 2)
        )
    )

# Display Results
print(
    f"{'Model':<10}"
    f"{'Layers':<10}"
    f"{'Neurons':<20}"
    f"{'Activation':<12}"
    f"{'Epochs':<10}"
    f"{'Accuracy'}"
)

for result in results:
    print(
        f"{result[0]:<10}"
        f"{result[1]:<10}"
        f"{str(result[2]):<20}"
        f"{result[3]:<12}"
        f"{result[4]:<10}"
        f"{result[5]}%"
    )