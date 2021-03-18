# -*- coding: utf-8 -*-

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Classifying Reuters newswires: a MULTICLASS classification example

# Loading Reuters newswires dataset

data_source = reuters.load_data(num_words=10000)
train_data, train_labels = data_source[0][0], data_source[0][1]
test_data, test_labels = data_source[1][0], data_source[1][1]


# ====================
# Preparing the data
# ====================

# encoding the data
def vectorize_sequences(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# one-hot encode labels
onehot_train_labels = to_categorical(train_labels)
onehot_test_labels = to_categorical(test_labels)

# set train, test and validation sets
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

x_val = x_train[:1000]
y_val = onehot_train_labels[:1000]

partial_x_train = x_train[1000:]
partial_y_train = onehot_train_labels[1000:]

# =================
# Model definition
# =================

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# Compiling the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# ================================================
# Plotting train and validation loss and accuracy
# ================================================

# train and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.clf()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Train and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# train and validation accuracy
plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Train and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =====================
# evaluate on test set
# =====================

results = model.evaluate(x_test, onehot_test_labels)

# =====================================
# set a random baseline classification
# =====================================

test_labels_copy = test_labels.copy()
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
accuracy_random = float(np.sum(hits_array)) / len(test_labels)

# ===================================
# Generating predictions on new data
# ===================================

predict_prob = model.predict(x_test)
predict_class = model.predict_classes(x_test)
