import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import models
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import importing as i


def build_DNN(input_file, test_split=0.2, random=1, classes_num=2):
    X, y = i.import_profiling_results(input_file)
    X = np.array(X, dtype=np.float32)
    y = np.eye(classes_num)[y]

    input_dim = X.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random)

    # model
    model = models.Sequential([
        layers.Input((input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(classes_num, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.25,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save("./dnn.keras")

    # evaluation
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n=== Final Test Metrics ===")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    plt.figure(figsize=(12, 5))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Izguba na učni množici')
    plt.plot(history.history['val_loss'], label='Izguba pri validacijski množici')
    plt.title('Izguba glede na iteracijo učenja')
    plt.xlabel('Epoha (iteracija učenja)')
    plt.ylabel('Izguba (0-1)')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Natančnost na učni množici')
    plt.plot(history.history['val_accuracy'], label='Natančnost na validacijski množici')
    plt.title('Natančnost glede na iteracijo učenja')
    plt.xlabel('Epoha (iteracija učenja)')
    plt.ylabel('Natančnost')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    
    # Confusion matrix
    y_pred_prob = model.predict(x_test)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, normalize="pred")
    class_names = np.arange(classes_num)

    # Plot confusion matrix
    plt.figure(figsize=(12, 9))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Matrika zamenjav')
    plt.ylabel('Dejansko')
    plt.xlabel('Napovedano')
    plt.yticks(np.arange(classes_num), class_names)
    plt.xticks(np.arange(classes_num), class_names, rotation='vertical')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return model