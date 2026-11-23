
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR
    / "asl_alphabet_dataset"
    / "asl_alphabet_train"
    / "asl_alphabet_train"
)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="int"
)

# Load validation dataset
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="int"
)

class_names = train_data.class_names

print("\nClasses:", class_names)
print("Total classes:", len(class_names))

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)

# Build CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(64, 64, 3)),

    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Stop training when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Save model
model.save(BASE_DIR / "sign_language_cnn_29.keras")

# Save class labels
with open(BASE_DIR / "class_names.txt", "w") as file:
    file.write("\n".join(class_names))

print("\nTraining completed!")
print("Model saved: sign_language_cnn_29.keras")
print("Class names saved: class_names.txt")
