# Assignment 1 ICS661 - Juliette Raubolt
# MLP model key features: 
# 1. Integer labels**
# 2. Input layer (with the 784 features), 
#    three** hidden layers (128 and 64 nodes, 32 nodes), 
#    and output layer (10 nodes for 10 labels)
# 3. Activation functions: relu for hidden layers, softmax for output layer
# 4. Optimizer: adam
# 5. Loss function: sparse_categorical_crossentropy
# 6. Metrics: accuracy, precision, recall (precision and recall calculated manually)
# 7. Early stopping with patience of 5
# 8. Batch size: 32
# 9. Validation split: 0.2
# 10. Epochs: 100
#
# **Indicates key feature that differs from other models in this series


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

# Load the data
train_data = np.loadtxt('train.csv', delimiter=',')
test_data = np.loadtxt('test.csv', delimiter=',')

# Split the data into input (features) and output (labels)
X_train = train_data[:, 1:]  # All rows, columns 1 to 784 (features)
y_train = train_data[:, 0]   # All rows, column 0 (labels)

X_test = test_data[:, 1:]    # All rows, columns 1 to 784 (features)
y_test = test_data[:, 0]     # All rows, column 0 (labels)

# Create the model
model = Sequential()
model.add(Input(shape=(784,)))  # Input layer for 784 features
model.add(Dense(128, activation='relu'))  # First hidden layer
model.add(Dense(64, activation='relu'))   # Second hidden layer
model.add(Dense(32, activation='relu'))   # Third hidden layer
model.add(Dense(10, activation='softmax'))  # Output layer for 10 classes

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model *set verbose to 0 when saving output to file
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')

# Predict the labels and calculate precision/recall manually
y_pred = np.argmax(model.predict(X_test), axis=1)
test_precision = precision_score(y_test, y_pred, average='macro')
test_recall = recall_score(y_test, y_pred, average='macro')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')

# Calculate F1 Score based on precision and recall
if(test_precision + test_recall == 0):
    f1_score = 0
else:
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
print(f'F1 Score: {f1_score}')

# Plot the training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()