import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

# Load the dataset
data = pd.read_csv('data2.csv')

# Preprocessing and Standardization
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_data = data[:int(data.shape[0] * 0.8), :]
test_data = data[int(data.shape[0] * 0.8):, :]

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=50, batch_size=32, validation_data=(test_data, test_labels))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test Accuracy: ', test_acc)