import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import streamlit_shadcn_ui as ui

# Load the dataset
df = pd.read_csv('/Users/raeesaparsaad/ITDAA/heart.csv', delimiter=';')

try:
    historical_data = pd.read_csv('/Users/raeesaparsaad/ITDAA/heart.csv', delimiter=';')
except FileNotFoundError:
    st.error("Historical data file not found. Please ensure 'historical_data.csv' is present in the directory.")
    st.stop()


# Define the features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
model = SVC(probability=True)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model and scaler to disk
joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the trained model and scaler
try:
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found/saved. Please ensure 'svm_model.pkl' and 'scaler.pkl' are present in the directory.")
    st.stop()

# Create the Streamlit application & Function to render the Patient details page
def patient_details_page():
    st.title('Heart Disease Prediction Web App')
    st.subheader('Fill in the form below to determine further treatment')
    
    # Input fields for patient details
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
    restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # Map input fields to numerical values
    sex_mapping = {'Male': 1, 'Female': 0}
    cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_mapping = {'True': 1, 'False': 0}
    restecg_mapping = {'Normal': 0, 'Having ST-T wave abnormality': 1, 'Showing probable or definite left ventricular hypertrophy': 2}
    exang_mapping = {'Yes': 1, 'No': 0}
    slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_mapping = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_mapping[sex]],
        'cp': [cp_mapping[cp]],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs_mapping[fbs]],
        'restecg': [restecg_mapping[restecg]],
        'thalach': [thalach],
        'exang': [exang_mapping[exang]],
        'oldpeak': [oldpeak],
        'slope': [slope_mapping[slope]],
        'ca': [ca],
        'thal': [thal_mapping[thal]]
    })
    st.session_state['input_data'] = input_data

    # Button to trigger prediction
    if st.button('Predict'):
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        prediction_proba = model.predict_proba(input_data_scaled)[0][1]

        # Display prediction result
        st.write("Patient details submitted successfully!")
        if prediction == 1:
            st.success(f'Prediction: Patient likely has heart disease with a probability of  {prediction_proba:.2f}.', icon="🚨")
            return input_data
        else:
            st.success(f'Prediction: Patient likely does not have heart disease with a probability of {prediction_proba:.2f}.', icon="✅")
            return input_data


# Display input data distribution compared to historical data
def results_page():
    # if 'patient_data' in st.session_state:
        st.write("## Input Data Distribution vs. Historical Data")
        means = historical_data.mean()
        df1 = st.session_state['input_data']
        input_data_mean = df1.iloc[0]

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(means.index, means.values, label='Historical Mean', marker='o')
        ax.plot(input_data_mean.index, input_data_mean.values, label='Input Data', marker='o')
        ax.set_xticks(range(len(means.index)))
        ax.set_xticklabels(means.index, rotation=45)
        ax.set_title('Comparison of Patient Input Data with Historical Mean')
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        ax.legend()

        st.pyplot(fig)
    # else:
        # st.write("Please enter patient details first.")

# Create tabs using radio buttons
tab = ui.tabs(options=['Patient details', 'Results'], default_value='Patient details', key="kanaries")
# tab = st.radio(
#     label='Select a tab',
#     options=['Patient details', 'Results'],
#     index=0,  # Default to 'Patient details'
#     key='kanaries'
# )

# Function to render the Patient details page
# def patient_details_page():
#     st.title("Patient Details")
#     st.write("Enter the details of the patient.")
#     # Add form or input fields for patient details here
#     name = st.text_input("Name")
#     age = st.number_input("Age", min_value=0, max_value=120, step=1)
#     blood_pressure = st.number_input("Resting Blood Pressure")
#     cholesterol = st.number_input("Cholesterol mg/dl")
#     max_heart_rate = st.number_input("Max Heart Rate")
#     st_depression = st.number_input("ST Depression")
#     submit_button = st.button("Submit")
#     if submit_button:
#         st.session_state.patient_data = {
#             "name": name,
#             "age": age,
#             "blood_pressure": blood_pressure,
#             "cholesterol": cholesterol,
#             "max_heart_rate": max_heart_rate,
#             "st_depression": st_depression
#         }
#         st.write("Patient details submitted successfully!")

# # Function to render the Results page
# def results_page():
#     st.title("Results")
#     if 'patient_data' in st.session_state:
#         st.write("Here are the results based on the input data.")
#         st.write(st.session_state.patient_data)
#         # Add code to display results here
#         # For example, predict heart disease based on patient data
#     else:
#         st.write("Please enter patient details first.")

# Render the appropriate page based on the selected tab
if tab == 'Patient details':
    patient_details_page()
elif tab == 'Results':
    results_page()







    