import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 48
# flow_from_directory sorts the class folders alphabetically
EMOTIONS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]


def plot_model_history(model_history, path='plot.png'):
    """Plot the accuracy and loss curves of a training run"""
    history = model_history.history
    epochs = range(1, len(history['accuracy']) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric, title in ((axs[0], 'accuracy', 'Model Accuracy'), (axs[1], 'loss', 'Model Loss')):
        ax.plot(epochs, history[metric])
        ax.plot(epochs, history['val_' + metric])
        ax.set_title(title)
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(['train', 'val'], loc='best')
    fig.savefig(path)
    plt.close(fig)


def build_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(EMOTIONS), activation='softmax'))
    return model


def make_generator(directory, batch_size, shuffle=True):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        directory,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=shuffle)


def train(args):
    train_generator = make_generator(os.path.join(args.data, 'train'), args.batch_size)
    validation_generator = make_generator(os.path.join(args.data, 'test'), args.batch_size, shuffle=False)

    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    model_info = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator)
    # Save the weights first so a plotting problem can not lose them
    model.save_weights(args.weights)
    plot_model_history(model_info)


def display(args):
    """Show the predicted emotion on each face in the webcam feed"""
    model = build_model()
    model.load_weights(args.weights)

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # Find haar cascade to draw bounding box around face (shipped with opencv-python)
    facecasc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    if facecasc.empty():
        raise SystemExit("Could not load the face cascade")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (IMAGE_SIZE, IMAGE_SIZE)), -1), 0)
            # Training images are rescaled by 1/255, so the webcam crop must be too
            prediction = model.predict(cropped_img / 255.0, verbose=0)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, EMOTIONS[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train the emotion CNN or run it on the webcam")
    ap.add_argument("--mode", choices=["train", "display"], required=True)
    ap.add_argument("--data", default="data", help="folder with train/ and test/ (default: data)")
    ap.add_argument("--weights", default="model.h5", help="weights file to save or load (default: model.h5)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "train":
        train(args)
    else:
        display(args)


if __name__ == "__main__":
    main()
