import pandas as pd
#import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import streamlit_shadcn_ui as ui
import os

class HeartDiseasePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.historical_data = None
        self.model = None
        self.scaler = None
        self.load_data()
        self.train_model()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path, delimiter=';')
            self.historical_data = pd.read_csv(self.data_path, delimiter=';')
        except FileNotFoundError:
            st.error("Historical data file not found. Please ensure the data file is present in the directory.")
            st.stop()

    def filter_target_1(self, target):
        return self.df.loc[self.df['target'] == target]

    def train_model(self):
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = SVC(probability=True)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")
        
        joblib.dump(self.model, 'svm_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')

    def load_model(self):
        try:
            self.model = joblib.load('svm_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            st.error("Model or scaler file not found/saved. Please ensure 'svm_model.pkl' and 'scaler.pkl' are present in the directory.")
            st.stop()

    def patient_details_page(self):
        st.title('Heart Disease Prediction Web App')
        st.subheader('Fill in the form below to determine further treatment')
        
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

        if st.button('Predict'):
            input_data_scaled = self.scaler.transform(input_data)
            prediction = self.model.predict(input_data_scaled)[0]
            prediction_proba = self.model.predict_proba(input_data_scaled)[0][1]

            st.write("Patient details submitted successfully!")
            if prediction == 1:
                st.success(f'Prediction: Patient likely has heart disease with a probability of  {prediction_proba:.2f}.', icon="🚨")
            else:
                st.success(f'Prediction: Patient likely does not have heart disease with a probability of {prediction_proba:.2f}.', icon="✅")

    def results_page(self):
        st.write("## Input Data Distribution vs. Historical Data")
        means = self.historical_data.mean()
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

    def run(self):
        tab = ui.tabs(options=['Patient details', 'Results'], default_value='Patient details', key="kanaries")
        if tab == 'Patient details':
            self.patient_details_page()
        elif tab == 'Results':
            self.results_page()

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), 'heart.csv')
    predictor = HeartDiseasePredictor(data_path)
    predictor.run()
