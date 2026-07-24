import pandas as pd
import streamlit as st
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("🫀Heart Disease Prediction")
st.write("Enter the patient's details below and click Predict.")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=45
)

restingbp = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=700,
    value=200
)

fastingbs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

maxhr = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chestpain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

restingecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

exerciseangina = st.selectbox(
    "Exercise Angina",
    ["N", "Y"]
)

stslope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [age],
        "RestingBP": [restingbp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fastingbs],
        "MaxHR": [maxhr],
        "Oldpeak": [oldpeak],
        "Sex": [sex],
        "ChestPainType": [chestpain],
        "RestingECG": [restingecg],
        "ExerciseAngina": [exerciseangina],
        "ST_Slope": [stslope]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(columns=columns, fill_value=0)

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("❤️ Heart Disease: YES")
    else:
        st.success("🤍 Heart Disease: NO")



# import streamlit as st
# import pandas as pd
# import joblib

# # This must be the first Streamlit command
# st.set_page_config(
#     page_title="🫀Heart Disease Prediction",
#     layout="centered"
# )

# st.write("1. Starting app")

# try:
#     model = joblib.load("heart_model.pkl")
#     st.write("2. Model loaded")

#     scaler = joblib.load("scaler.pkl")
#     st.write("3. Scaler loaded")

#     columns = joblib.load("columns.pkl")
#     st.write("4. Columns loaded")

# except Exception as e:
#     st.error(f"Error loading files: {e}")
#     st.stop()

# st.title("🫀 Heart Disease Prediction")
# st.write("Enter the patient's details below and click Predict.")

# age = st.number_input("Age", min_value=1, max_value=120, value=45)
# restingbp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
# cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
# fastingbs = st.selectbox("Fasting Blood Sugar", [0, 1])
# maxhr = st.number_input("Maximum Heart Rate", min_value=50, max_value=250, value=150)
# oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# sex = st.selectbox("Sex", ["M", "F"])
# chestpain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
# restingecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
# exerciseangina = st.selectbox("Exercise Angina", ["N", "Y"])
# stslope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# if st.button("Predict"):
#     try:
#         input_data = pd.DataFrame({
#             "Age": [age],
#             "RestingBP": [restingbp],
#             "Cholesterol": [cholesterol],
#             "FastingBS": [fastingbs],
#             "MaxHR": [maxhr],
#             "Oldpeak": [oldpeak],
#             "Sex": [sex],
#             "ChestPainType": [chestpain],
#             "RestingECG": [restingecg],
#             "ExerciseAngina": [exerciseangina],
#             "ST_Slope": [stslope]
#         })

#         input_data = pd.get_dummies(input_data)
#         input_data = input_data.reindex(columns=columns, fill_value=0)
#         input_data = scaler.transform(input_data)

#         prediction = model.predict(input_data)

#         if prediction[0] == 1:
#             st.error("❤️ Heart Disease: YES")
#         else:
#             st.success("💚 Heart Disease: NO")

#     except Exception as e:
#         st.error(f"Prediction Error: {e}")