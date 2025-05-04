import json
from ngram_profile import FullProfiler
import numpy as np
import tensorflow as tf
import keras
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical


def build_DNN(profiler, labeled_sets, input_dim=3, classes_num=2):
    X = []
    y = []

    # creating list of learning data
    labels = list(labeled_sets.keys())
    data = []

    for label_idx, label in enumerate(labels):
        numbers = labeled_sets[label]
        first_key = next(iter(numbers))
        for v in numbers[first_key]:
            data.append(v)
            
    
    feature_vector = []        
    for item in data:
        feature_vector.append(profiler.compare(item))
    X.append(feature_vector)
    y.append(label_idx)

    X = np.array(X)
    y = np.eye(classes_num)[y]
    model = models.Sequential()
    model.add(layers.Input((input_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(classes_num, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=20, batch_size=8, verbose=1)

    return model