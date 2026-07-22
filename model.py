import pandas as pd
import streamlit as st
import joblib

model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")
scaler = joblib.load("scaler.pkl")

sample = pd.DataFrame({
    'Age': [45],
    'RestingBP': [130],
    'Cholesterol': [230],
    'FastingBS': [0],
    'MaxHR': [150],
    'Oldpeak': [1.5],
    'Sex': ['M'],
    'ChestPainType': ['ATA'],
    'RestingECG': ['Normal'],
    'ExerciseAngina': ['N'],
    'ST_Slope': ['Up']
})

sample = pd.get_dummies(sample)

sample = sample.reindex(columns=columns, fill_value=0)

sample = scaler.transform(sample)

# Make prediction
prediction = model.predict(sample)

# Print result
if prediction[0] == 1:
    print("Prediction: Heart Disease Detected")
else:
    print("Prediction: No Heart Disease")