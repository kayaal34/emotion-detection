# 🧠 Emotion Detection & Training Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.3-orange)

This project is a Deep Learning pipeline designed to process raw image data and train a Convolutional Neural Network (CNN) for emotion classification.

It automatically organizes raw pixel data into categorized folders (Angry, Happy, Sad, etc.) and performs model training on this structured dataset.

## ⚙️ Key Features
* **Data Organization:** Converts raw CSV pixel data into a structured image directory system (`train` and `test` folders).
* **Deep Learning Training:** Trains a custom CNN model using TensorFlow/Keras to recognize 7 different emotions.
* **Accuracy & Loss Plotting:** Generates performance graphs after training.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KULLANICIADIN/emotion-detection.git](https://github.com/KULLANICIADIN/emotion-detection.git)
    cd emotion-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run

### Step 1: Prepare the Dataset
First, organize the raw data into emotion-specific folders:
```bash
python create_dataset.py
This will create a data/ folder with sorted images.

Step 2: Train the Model
Train the CNN model using the processed images:

Bash

python emotion_recognition.py --mode train
The trained model weights will be saved as model.h5 upon completion.
