import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# FER2013 label order
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
IMAGE_SIZE = 48
# FER2013 has 28709 training rows; used when the csv has no "Usage" column
TRAIN_ROWS = 28709


def pixels_to_image(pixels):
    """Convert the space separated pixel string of one row to a 48x48 image"""
    values = np.array(pixels.split(), dtype=np.uint8)
    if values.size != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"expected {IMAGE_SIZE * IMAGE_SIZE} pixels, got {values.size}")
    return Image.fromarray(values.reshape(IMAGE_SIZE, IMAGE_SIZE))


def split_for(index, usage=None):
    """Return "train" or "test" for a row"""
    if isinstance(usage, str):
        return 'train' if usage == 'Training' else 'test'
    return 'train' if index < TRAIN_ROWS else 'test'


def make_folders(out_dir):
    for split in ('train', 'test'):
        for emotion in EMOTIONS:
            os.makedirs(os.path.join(out_dir, split, emotion), exist_ok=True)


def prepare(df, out_dir='data'):
    """Save every row of the dataframe as a png in out_dir/<split>/<emotion>/"""
    make_folders(out_dir)
    counts = {}
    usages = df['Usage'] if 'Usage' in df.columns else [None] * len(df)

    for i, (emotion, pixels, usage) in enumerate(tqdm(zip(df['emotion'], df['pixels'], usages), total=len(df))):
        split = split_for(i, usage)
        name = EMOTIONS[int(emotion)]
        key = (split, name)
        number = counts.get(key, 0)
        pixels_to_image(pixels).save(os.path.join(out_dir, split, name, f'im{number}.png'))
        counts[key] = number + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="fer2013.csv", help="path to fer2013.csv")
    ap.add_argument("--out", default="data", help="output folder")
    args = ap.parse_args()

    print("Saving images...")
    counts = prepare(pd.read_csv(args.csv), args.out)
    for (split, name), number in sorted(counts.items()):
        print(f"{split}/{name}: {number}")
    print("Done!")


if __name__ == '__main__':
    main()
