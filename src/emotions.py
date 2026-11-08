import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
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


def conv_block(model, filters):
    """Two 3x3 convolutions with batch normalization, then pooling and dropout"""
    for _ in range(2):
        model.add(Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


def build_model():
    model = Sequential()
    model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    # 48x48 -> 24x24 -> 12x12 -> 6x6
    for filters in (32, 64, 128):
        conv_block(model, filters)

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(len(EMOTIONS), activation='softmax'))
    return model


def make_generator(directory, batch_size, shuffle=True, augment=False):
    if augment:
        # Small random changes so the model does not memorise the training faces
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        directory,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=shuffle)


def train(args):
    train_generator = make_generator(os.path.join(args.data, 'train'), args.batch_size,
                                     augment=not args.no_augment)
    validation_generator = make_generator(os.path.join(args.data, 'test'), args.batch_size, shuffle=False)

    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=args.lr), metrics=['accuracy'])
    callbacks = [
        # Keep the weights of the best epoch, not the last one
        ModelCheckpoint(args.weights, monitor='val_accuracy', save_best_only=True,
                        save_weights_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
    ]
    model_info = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=callbacks)
    plot_model_history(model_info)
    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"Validation accuracy: {accuracy:.3f} (loss {loss:.3f}), weights saved to {args.weights}")


# BGR box colour for each emotion
COLORS = {
    "Angry": (0, 0, 255), "Disgusted": (0, 128, 0), "Fearful": (128, 0, 128),
    "Happy": (0, 215, 255), "Neutral": (200, 200, 200), "Sad": (255, 0, 0),
    "Surprised": (0, 165, 255),
}


def load_face_detector():
    # Haar cascade shipped with opencv-python
    facecasc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    if facecasc.empty():
        raise SystemExit("Could not load the face cascade")
    return facecasc


def preprocess_faces(gray, faces):
    """Crop, resize and scale the faces like the training images (N, 48, 48, 1)"""
    crops = [cv2.resize(gray[y:y + h, x:x + w], (IMAGE_SIZE, IMAGE_SIZE)) for (x, y, w, h) in faces]
    return np.expand_dims(np.array(crops, dtype=np.float32) / 255.0, -1)


def predict_faces(model, facecasc, frame):
    """Return (box, emotion, confidence) for every face in a BGR frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return []
    # One predict call for all faces instead of one per face
    predictions = model.predict(preprocess_faces(gray, faces), verbose=0)
    return [(tuple(int(v) for v in box), EMOTIONS[int(np.argmax(p))], float(np.max(p)))
            for box, p in zip(faces, predictions)]


def draw_results(frame, results):
    for (x, y, w, h), emotion, confidence in results:
        color = COLORS[emotion]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{emotion} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        top = max(y - th - 10, 0)
        cv2.rectangle(frame, (x, top), (x + tw + 10, top + th + 10), color, -1)
        cv2.putText(frame, label, (x + 5, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def display(args):
    """Show the predicted emotion on each face in the webcam feed"""
    model = build_model()
    model.load_weights(args.weights)
    facecasc = load_face_detector()

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Mirror the image so it moves like a mirror
        frame = cv2.flip(frame, 1)
        draw_results(frame, predict_faces(model, facecasc, frame))
        cv2.imshow('Emotion detection (q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_image(args):
    """Print the emotions found in a photo and save a copy with the boxes drawn"""
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read {args.image}")
    model = build_model()
    model.load_weights(args.weights)
    results = predict_faces(model, load_face_detector(), frame)
    if not results:
        print("No face found")
        return
    for (x, y, w, h), emotion, confidence in results:
        print(f"Face at ({x}, {y}, {w}x{h}): {emotion} {confidence:.0%}")
    root, ext = os.path.splitext(args.image)
    out = f"{root}_emotions{ext}"
    cv2.imwrite(out, draw_results(frame, results))
    print(f"Saved {out}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train the emotion CNN or run it on the webcam or a photo")
    ap.add_argument("--mode", choices=["train", "display", "image"], required=True)
    ap.add_argument("--image", help="photo to analyse in image mode")
    ap.add_argument("--data", default="data", help="folder with train/ and test/ (default: data)")
    ap.add_argument("--weights", default="model.h5", help="weights file to save or load (default: model.h5)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)")
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many epochs without a better val_loss (default: 8)")
    ap.add_argument("--no-augment", action="store_true", help="train without data augmentation")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "image" and not args.image:
        raise SystemExit("--mode image needs --image path/to/photo.jpg")
    if args.mode == "train":
        train(args)
    elif args.mode == "image":
        predict_image(args)
    else:
        display(args)


if __name__ == "__main__":
    main()
