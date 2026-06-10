
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("./models/digit_model.h5")


def predict_digit():

    file_path = filedialog.askopenfilename(
        title="Select Digit Image",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg")
        ]
    )

    if not file_path:
        return

    image = Image.open(file_path)

    image = image.convert("L")

    image = image.resize((28, 28))

    image_array = np.array(image)

    image_array = 255 - image_array

    image_array = image_array / 255.0

    image_array = image_array.reshape(
        1,
        28,
        28
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_digit = np.argmax(prediction)

    confidence_score = np.max(prediction) * 100

    result_label.config(
        text=
        f"Predicted Digit : {predicted_digit}\n"
        f"Confidence Score : {confidence_score:.2f}%"
    )


root = tk.Tk()

root.title("Digit Recognition App")
root.geometry("450x300")

heading = tk.Label(
    root,
    text="Handwritten Digit Prediction",
    font=("Arial", 16, "bold")
)

heading.pack(pady=20)

upload_button = tk.Button(
    root,
    text="Upload Digit Image",
    command=predict_digit,
    font=("Arial", 12)
)

upload_button.pack(pady=20)

result_label = tk.Label(
    root,
    text="Upload an image to predict",
    font=("Arial", 12)
)

result_label.pack(pady=20)

root.mainloop()