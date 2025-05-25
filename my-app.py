import streamlit as st  # Creates the web application interface
import pandas as pd   # Data manipulation and structuring
import numpy as np  # Handles all numerical data transformations
import joblib  # Saves and loads models 
import gdown  # Google Drive file downloader
import os  # Checks if model file exists locally (os.path.exists)
from sklearn.preprocessing import LabelEncoder  # Converts text categories (like "Male"/"Female") to numerical values

# Google Drive file ID of the model
GDRIVE_FILE_ID = "13551x43fiENQqSjpC7oXDRtQsAF-BNJg"
MODEL_PATH = "heart_attack_risk_model.pkl"

# Function to download the model from Google Drive
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

# Load the trained model
model = download_model()

# Create label encoders for categorical features
# Each encoder converts text categories to numerical values for the model

@st.cache_resource
def get_label_encoders():
    encoders = {
        'Sex': LabelEncoder().fit(['Male', 'Female']),
        'Diet': LabelEncoder().fit(['Healthy', 'Average', 'Unhealthy']),
        'Country': LabelEncoder().fit([
            'Argentina', 'Canada', 'France', 'Thailand', 'Germany', 'Japan', 'Brazil', 
            'South Africa', 'United States', 'India', 'Spain', 'Vietnam', 'China', 
            'Nigeria', 'New Zealand', 'Australia', 'United Kingdom', 'South Korea', 
            'Italy', 'Colombia'
        ]),
        'Continent': LabelEncoder().fit([
            'South America', 'North America', 'Europe', 'Asia', 'Africa', 'Australia'
        ]),
        'Hemisphere': LabelEncoder().fit(['Southern Hemisphere', 'Northern Hemisphere'])
    }
    return encoders

encoders = get_label_encoders()


# Function to preprocess input data
def preprocess_input(input_data):
    # Convert categorical features using label encoders
    input_data['Sex'] = encoders['Sex'].transform([input_data['Sex']])[0]
    input_data['Diet'] = encoders['Diet'].transform([input_data['Diet']])[0]
    input_data['Country'] = encoders['Country'].transform([input_data['Country']])[0]
    input_data['Continent'] = encoders['Continent'].transform([input_data['Continent']])[0]
    input_data['Hemisphere'] = encoders['Hemisphere'].transform([input_data['Hemisphere']])[0]
    
#  Sex: "Male" → 0, "Female" → 1

# Diet: "Healthy" → 0, "Average" → 1, "Unhealthy" → 2


    # Create DataFrame with all expected columns
    features = [
        'Patient ID','Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 
        'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption',
        'Exercise Hours Per Week', 'Diet', 'Previous Heart Problems', 
        'Medication Use', 'Stress Level', 'Sedentary Hours Per Day', 
        'Income', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week',
        'Sleep Hours Per Day', 'Country', 'Continent', 'Hemisphere',
        'Systolic_BP', 'Diastolic_BP'  
    ]
    
    # Add Patient ID as numeric 0 (won't affect prediction)
    input_data['Patient ID'] = 0
    
    # Ensure all columns are in the correct order
    df = pd.DataFrame([input_data], columns=features)
    return df

# Main app function
def main():
    st.title("Heart Attack Risk Prediction")
    st.write("""
    This app predicts the risk of heart attack based on health metrics and lifestyle factors.
    Please fill in your information below.
    """)
    
    with st.form("patient_info"):
        st.header("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            country = st.selectbox("Country", [
                "Argentina", "Canada", "France", "Thailand", "Germany", "Japan", 
                "Brazil", "South Africa", "United States", "India", "Spain", 
                "Vietnam", "China", "Nigeria", "New Zealand", "Australia", 
                "United Kingdom", "South Korea", "Italy", "Colombia"
            ])
            continent = st.selectbox("Continent", [
                "South America", "North America", "Europe", "Asia", "Africa", "Australia"
            ])
            hemisphere = st.selectbox("Hemisphere", [
                "Southern Hemisphere", "Northern Hemisphere"
            ])
            diet = st.selectbox("Diet", ["Healthy", "Average", "Unhealthy"])
            
        with col2:
            systolic_bp = st.number_input("Systolic Blood Pressure", min_value=90, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=60, max_value=120, value=80)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=30, max_value=800, value=150)
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
            heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=72)
        
        st.header("Medical History")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            family_history = st.selectbox("Family History of Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            previous_heart_problems = st.selectbox("Previous Heart Problems", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col2:
            smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            medication_use = st.selectbox("Medication Use", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col3:
            alcohol_consumption = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            physical_activity = st.slider("Physical Activity Days Per Week", 0, 7, 3)
        
        st.header("Lifestyle Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            exercise_hours = st.number_input("Exercise Hours Per Week", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
            sedentary_hours = st.number_input("Sedentary Hours Per Day", min_value=0.0, max_value=16.0, value=6.0, step=0.5)
            
        with col2:
            sleep_hours = st.number_input("Sleep Hours Per Day", min_value=4, max_value=12, value=7)
            income = st.number_input("Annual Income (USD)", min_value=10000, max_value=300000, value=100000, step=1000)
            
        submitted = st.form_submit_button("Predict Heart Attack Risk")
    
    if submitted and model is not None:
        # Create input dictionary
        input_data = {
            'Age': age,
            'Sex': sex,
            'Cholesterol': cholesterol,
            'Heart Rate': heart_rate,
            'Diabetes': diabetes,
            'Family History': family_history,
            'Smoking': smoking,
            'Obesity': obesity,
            'Alcohol Consumption': alcohol_consumption,
            'Exercise Hours Per Week': exercise_hours,
            'Diet': diet,
            'Previous Heart Problems': previous_heart_problems,
            'Medication Use': medication_use,
            'Stress Level': stress_level,
            'Sedentary Hours Per Day': sedentary_hours,
            'Income': income,
            'BMI': bmi,
            'Triglycerides': triglycerides,
            'Physical Activity Days Per Week': physical_activity,
            'Sleep Hours Per Day': sleep_hours,
            'Country': country,
            'Continent': continent,
            'Hemisphere': hemisphere,
            'Systolic_BP': systolic_bp,  # Using the separate BP measurements
            'Diastolic_BP': diastolic_bp  # Using the separate BP measurements
        }
        
        # Preprocess and predict
        processed_data = preprocess_input(input_data)
        
        try:
            # Ensure the columns are in the exact same order as during training
            expected_columns = [
                 'Patient ID' ,'Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 
                'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption',
                'Exercise Hours Per Week', 'Diet', 'Previous Heart Problems', 
                'Medication Use', 'Stress Level', 'Sedentary Hours Per Day', 
                'Income', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week',
                'Sleep Hours Per Day', 'Country', 'Continent', 'Hemisphere',
                'Systolic_BP', 'Diastolic_BP'
            ]
            processed_data = processed_data[expected_columns]
            
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction[0] == 1:
                st.error(f"High Risk of Heart Attack (Probability: {prediction_proba[0][1]:.2%})")
                st.write("""
                *Recommendations:*
                - Consult with a cardiologist immediately
                - Adopt a heart-healthy diet
                - Increase physical activity
                - Reduce stress levels
                - Monitor blood pressure and cholesterol regularly
                - Quit smoking if applicable
                - Limit alcohol consumption
                """)
            else:
                st.success(f"Low Risk of Heart Attack (Probability: {prediction_proba[0][1]:.2%})")
                
               # .2% ==== 0.752389 → 75.24% (rounded to 2 decimal places)
               
                st.write("""
                *Recommendations to Maintain Heart Health:*
                - Continue healthy lifestyle habits
                - Regular check-ups with your doctor
                - Maintain balanced diet and exercise routine
                - Manage stress effectively
                - Monitor key health indicators regularly
                - Maintain healthy sleep patterns
                """)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("""
            *Troubleshooting Steps:*
            1. Verify all the input values are within expected ranges
            2. Check that all required fields are filled
            3. Ensure the model was trained with matching features
            """)
            # Debugging information
            st.write("Features being sent to model:", processed_data.columns.tolist())

if __name__ == "__main__":
    main()
