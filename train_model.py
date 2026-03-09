import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

# Load dataset
df = pd.read_csv('stroke_prediction_dataset.csv')

# Select features and target
# Map expected features from app.py to csv columns
# app.py inputs: Age, Hypertension, Heart Disease, Average Glucose Level, BMI
feature_cols = ['Age', 'Hypertension', 'Heart Disease', 'Average Glucose Level', 'Body Mass Index (BMI)']
target_col = 'Diagnosis'

# Prepare data
X = df[feature_cols].copy()
y = df[target_col].apply(lambda x: 1 if x == 'Stroke' else 0)

# Handle missing values (simple drop for now as simple strategy)
# BMI is often missing in these datasets, but let's check input constraints
# app.py doesn't handle missing input, assumes valid numbers.
# We will drop NaNs to ensure clean training data.
data = pd.concat([X, y], axis=1).dropna()
X = data[feature_cols]
y = data[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Search for the pattern "scaler = pickle.load(open('scaler.pkl', 'rb'))" in app.py
# We must save it exactly as app.py expects
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Reshape for CNN (samples, time steps, features)
# app.py does: input_scaled = input_scaled.reshape(1, input_scaled.shape[1], 1) which implies (samples, features, 1)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Build CNN Model
# app.py treats it as a binary classification with sigmoid output
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('model.h5')
print("Model and scaler saved successfully.")
