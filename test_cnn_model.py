import tensorflow as tf
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sign_language_cnn_29.keras"
TEST_DIR = BASE_DIR / "asl_alphabet_dataset" / "asl_alphabet_test" / "asl_alphabet_test"
CLASS_FILE = BASE_DIR / "class_names.txt"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r") as file:
    class_names = file.read().splitlines()

correct = 0
total = 0

print("\nSIGN LANGUAGE CNN TEST RESULTS\n")

for image_path in sorted(TEST_DIR.glob("*")):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    actual_label = image_path.stem.split("_")[0]

    if actual_label not in class_names:
        continue

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(64, 64),
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    result = "CORRECT" if predicted_label == actual_label else "WRONG"

    if result == "CORRECT":
        correct += 1

    total += 1

    print(
        f"Actual: {actual_label} | "
        f"Predicted: {predicted_label} | "
        f"Confidence: {confidence:.2f}% | {result}"
    )

if total > 0:
    accuracy = (correct / total) * 100

    print("\n----------------------------")
    print(f"Total Images: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Wrong Predictions: {total - correct}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("----------------------------")