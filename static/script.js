
// ==========================================
// 1. CHECK JAVASCRIPT LOADING
// ==========================================

console.log("Sign Language JavaScript Loaded Successfully");


// ==========================================
// 2. GET HTML ELEMENTS
// ==========================================

const uploadInput = document.getElementById("uploadInput");
const uploadPreview = document.getElementById("uploadPreview");
const uploadPredictBtn = document.getElementById("uploadPredictBtn");

const uploadPrediction = document.getElementById("uploadPrediction");
const uploadConfidence = document.getElementById("uploadConfidence");
const uploadStatus = document.getElementById("uploadStatus");

let selectedImage = null;


// ==========================================
// 3. SELECT IMAGE
// ==========================================

uploadInput.addEventListener("change", function () {

    console.log("Image selection event triggered");

    const file = this.files[0];

    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {

        uploadStatus.textContent = "Please select a valid image.";

        uploadPredictBtn.disabled = true;

        return;
    }

    selectedImage = file;

    // Display image preview
    uploadPreview.src = URL.createObjectURL(file);

    uploadPreview.style.display = "block";

    // Reset previous prediction
    uploadPrediction.textContent = "?";

    uploadConfidence.textContent = "Confidence: --";

    uploadPredictBtn.disabled = false;

    uploadStatus.textContent =
        "Image uploaded successfully. Click Predict.";

    console.log("Selected image:", file.name);

});


// ==========================================
// 4. PREDICT UPLOADED IMAGE
// ==========================================

uploadPredictBtn.addEventListener("click", async function () {

    console.log("Predict button clicked");

    if (!selectedImage) {

        uploadStatus.textContent = "Please select an image first.";

        return;
    }

    const formData = new FormData();

    formData.append("image", selectedImage, selectedImage.name);

    uploadPredictBtn.disabled = true;

    uploadStatus.textContent = "Recognizing image...";

    console.log("Sending image to Flask CNN model...");

    try {

        // Send image to Flask
        const response = await fetch("/predict", {

            method: "POST",

            body: formData

        });

        console.log("Flask response status:", response.status);

        const result = await response.json();

        console.log("Prediction response:", result);

        if (!response.ok) {

            throw new Error(
                result.error || "Prediction failed."
            );

        }

        // Display prediction
        uploadPrediction.textContent = result.prediction;

        uploadConfidence.textContent =
            `Confidence: ${result.confidence}%`;

        uploadStatus.textContent =
            "Image recognition completed successfully!";

        console.log("Predicted sign:", result.prediction);

        console.log("Confidence:", result.confidence + "%");

    }

    catch (error) {

        uploadPrediction.textContent = "?";

        uploadConfidence.textContent = "Confidence: --";

        uploadStatus.textContent = error.message;

        console.error("Prediction error:", error);

    }

    finally {

        uploadPredictBtn.disabled = false;

    }

});
