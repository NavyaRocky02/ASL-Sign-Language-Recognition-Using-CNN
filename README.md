# Sign Language Recognition Using CNN

## Project Overview
This project uses Deep Learning and Convolutional Neural Networks (CNN) to recognize American Sign Language hand gestures from image data.

## Objectives
- Recognize hand signs from image pixels.
- Classify 24 different sign language classes.
- Train and evaluate a CNN model.
- Predict hand gestures using the trained model.

## Dataset
Sign Language MNIST Dataset.

- Training samples: 27,455
- Testing samples: 7,172
- Image dimensions: 28 × 28 pixels
- Number of classes: 24

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow
- Keras
- Scikit-learn

## Algorithm
Convolutional Neural Network (CNN).

## Project Structure
```text
Sign_Language_Recognition/
│
├── dataset/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
│
├── MNIST_sign.ipynb
├── requirements.txt
├── README.md
└── sign_language_cnn_model.keras
```

## How to Run
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Open the notebook in Jupyter.
4. Run all cells.

## Output
The trained CNN model predicts sign language classes from input images.

## Conclusion
This project demonstrates the use of deep learning for American Sign Language alphabet recognition.