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
    git clone https://github.com/kayaal34/emotion-detection.git
    cd emotion-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run

### Step 1: Prepare the Dataset
Download `fer2013.csv` (FER2013) and convert it into emotion folders:
```bash
cd src
python dataset_prepare.py --csv path/to/fer2013.csv
```
This creates `data/train/<emotion>/` and `data/test/<emotion>/` with 48x48 grayscale images.
The csv's `Usage` column decides the split (`Training` goes to train, the rest to test).

### Step 2: Train the Model
```bash
python emotions.py --mode train
```
The weights are saved as `model.h5` and the accuracy/loss curves as `plot.png`.

### Step 3: Run on the Webcam
```bash
python emotions.py --mode display
```
Press `q` to quit. The face detector uses the Haar cascade shipped with `opencv-python`.

## 🧪 Tests
The dataset preparation code has unit tests (they do not need TensorFlow):
```bash
pip install numpy pandas Pillow tqdm pytest
python -m pytest -q
```
