# 🧠 Emotion Detection & Training Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)

This project is a Deep Learning pipeline designed to process raw image data and train a Convolutional Neural Network (CNN) for emotion classification.

It automatically organizes raw pixel data into categorized folders (Angry, Happy, Sad, etc.) and performs model training on this structured dataset.

## ⚙️ Key Features
* **Data Organization:** Converts raw CSV pixel data into a structured image directory system (`train` and `test` folders).
* **Deep Learning Training:** Trains a CNN with batch normalization and data augmentation to recognize 7 different emotions. The best epoch is kept, the learning rate drops when validation stalls and training stops early when it stops improving.
* **Accuracy & Loss Plotting:** Generates performance graphs after training.
* **Webcam and photos:** Shows the emotion and its confidence on every detected face, live or on a saved photo.

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
The weights of the best epoch are saved as `model.weights.h5` and the accuracy/loss curves as `plot.png`.

| Option | Default | |
|---|---|---|
| `--epochs` | 50 | maximum number of epochs |
| `--batch-size` | 64 | |
| `--lr` | 0.0001 | learning rate |
| `--patience` | 8 | epochs without a better `val_loss` before stopping |
| `--no-augment` | off | train without data augmentation |
| `--data` | `data` | folder with `train/` and `test/` |
| `--weights` | `model.weights.h5` | weights file to save or load |

### Step 3: Run on the Webcam
```bash
python emotions.py --mode display
```
Press `q` to quit. Each face gets a box coloured by emotion and the model's confidence. Use `--camera 1` for a second webcam. The face detector uses the Haar cascade shipped with `opencv-python`.

### Or on a Photo
```bash
python emotions.py --mode image --image path/to/photo.jpg
```
Prints the emotion of each face and saves `photo_emotions.jpg` with the boxes drawn.

## 🧪 Tests
The dataset preparation code and the label order have unit tests that do not need TensorFlow:
```bash
pip install numpy pandas Pillow tqdm pytest
python -m pytest -q
```
With TensorFlow and OpenCV installed, the model tests run as well; otherwise they are skipped.
