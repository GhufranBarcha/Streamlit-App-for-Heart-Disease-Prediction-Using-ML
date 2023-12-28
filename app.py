import pandas as pd
import streamlit as st
import joblib

# Assuming you have already defined the 'pipeline' in your Jupyter code
pipeline = joblib.load("pipeline.pkl")
# Streamlit app
def predict_heart_disease():
    st.title("Heart Disease Prediction App")

    # User input form
    age = st.number_input("Enter the Age:", min_value=1, max_value=120, value=30)
    sex = st.radio("Enter the Sex:", ["M", "F"])
    chest_pain_type = st.selectbox("Enter the Chest Pain Type:", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Enter the Resting Blood Pressure:", min_value=1, value=120)
    cholesterol = st.number_input("Enter the Cholesterol level:", min_value=1, value=200)
    fasting_bs = st.radio("Fasting Blood Sugar (1 for > 120 mg/dl, 0 otherwise):", [0, 1])
    resting_ecg = st.selectbox("Resting ECG Type:", ["Normal", "ST", "Other"])
    max_hr = st.number_input("Enter the Maximum Heart Rate:", min_value=1, value=150)
    exercise_angina = st.radio("Exercise-Induced Angina (Yes, No):", ["N", "Y"])
    oldpeak = st.number_input("Enter the ST Depression induced by exercise relative to rest:", value=0.0)
    st_slope = st.selectbox("ST Slope Type:", ["Up", "Flat", "Down"])

    # Create a DataFrame with the user input
    data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # Ensure that the 'Sex' column is of type 'object' to match the encoder's expectations
    data['Sex'] = data['Sex'].astype('object')

    # Predict the data
    transformed_data = pipeline["Preprocessor"].transform(data)
    prediction = pipeline["classifier"].predict(transformed_data)

    # Display the prediction
    st.subheader("Prediction Result:")
    if prediction[0] == 0:
        st.success("No Heart Disease Detected.")
    else:
        st.error("Heart Disease Detected.")

if __name__ == "__main__":
    predict_heart_disease()
