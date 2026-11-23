
import os
import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from threading import Timer
import webbrowser

# ==========================================
# 1. FLASK INITIALIZATION
# ==========================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# 2. LOAD CNN MODEL
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "sign_language_cnn_29.keras"
)

CLASS_FILE = os.path.join(
    BASE_DIR,
    "class_names.txt"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"CNN model not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_FILE):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_FILE}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r") as file:
    class_names = file.read().splitlines()

print("CNN model loaded successfully")
print("Total classes:", len(class_names))


# ==========================================
# 3. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 4. IMAGE PREDICTION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image received"
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "error": "Please select an image"
            }), 400

        # Read uploaded image
        image_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "error": "Invalid image"
            }), 400

        # Convert BGR to RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize image to 64x64
        image = cv2.resize(
            image,
            (64, 64)
        )

        # Convert image to array
        image_array = image.astype(
            np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # CNN prediction
        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

        # Get predicted class
        predicted_index = np.argmax(
            probabilities
        )

        predicted_letter = class_names[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        # Top 3 predictions
        top_indices = np.argsort(
            probabilities
        )[::-1][:3]

        top_predictions = []

        for index in top_indices:

            top_predictions.append({

                "letter": class_names[index],

                "confidence": round(
                    float(probabilities[index] * 100),
                    2
                )

            })

        print("\n================================")
        print("Predicted sign:", predicted_letter)
        print("Confidence:", round(confidence, 2), "%")
        print("Top predictions:", top_predictions)
        print("================================")

        return jsonify({

            "prediction": predicted_letter,

            "confidence": round(
                confidence,
                2
            ),

            "top_predictions": top_predictions,

            "message": "Prediction completed successfully"

        })

    except Exception as error:

        print("Prediction error:", str(error))

        return jsonify({
            "error": str(error)
        }), 500


# ==========================================
# 5. OPEN BROWSER
# ==========================================

def open_browser():

    webbrowser.open_new(
        "http://127.0.0.1:5000"
    )


# ==========================================
# 6. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\n====================================")
    print("    SIGN LANGUAGE RECOGNITION")
    print("====================================")
    print("Algorithm: Convolutional Neural Network")
    print("Dataset: ASL Alphabet")
    print("Classes: 29")
    print("Mode: Image Upload")
    print("====================================")

    Timer(
        2,
        open_browser
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
