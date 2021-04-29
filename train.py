import pickle
import os

from skimage import io
from skimage import transform

from sklearn.linear_model import SGDClassifier

import numpy as np

def get_numpy_array_of_digit_from_filename(filename):
    digit_sample = io.imread(filename, as_gray=True)
    digit_sample = transform.resize(digit_sample, (100, 100)) # Resize it to a fixed size so we don't have problems when training the model.
    digit_sample = digit_sample.flatten() # We turn the digit_sample matrix into an unidimensional array.

    return digit_sample

def load_dataset():
    print("Importing dataset...")
    digits_folders = os.listdir("./dataset")

    dataset = []
    
    samples = []
    target_values = []
    for digit_folder_name in digits_folders:
        folder_path = "./dataset/" + digit_folder_name

        digit_samples_filenames = os.listdir(folder_path)
        for digit_sample_filename in digit_samples_filenames:
            digit_sample_path = folder_path + "/" + digit_sample_filename
            
            try:
                digit_sample = get_numpy_array_of_digit_from_filename(digit_sample_path)
                target_values.append(ord(digit_folder_name)) # We append the ascii code of the folder name, this is the value we expect from this sample.
                samples.append(digit_sample)
            except:
                continue
        
    dataset.append(samples)
    dataset.append(target_values)

    return dataset

dataset = load_dataset()
samples = dataset[0]
target_values = dataset[1]

print("Training model...")
clf = SGDClassifier(verbose=True)
clf.fit(samples, target_values)

print("Exporting model object into model.pkl...")
with open("./model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Training complete!")