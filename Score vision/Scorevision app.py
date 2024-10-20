import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv(r"C:\Users\kaeli\OneDrive - Richfield Graduate Institute of Technology\Courses\1st Year\Data science\Semseter 2\Project final\Score vision\StudentPerformanceFactors.csv")

# Define the feature and target variable for predicting the grades
X_grades = data.drop(columns=['Exam_Score'])
y_grades = data['Exam_Score']

# Identify categorical columns
categorical_cols = X_grades.select_dtypes(include=['object']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', X_grades.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split the dataset into training and testing sets for predicting grades
X_train_grades, X_test_grades, y_train_grades, y_test_grades = train_test_split(X_grades, y_grades, test_size=0.1, random_state=5981)

# Train the linear regression model
pipeline.fit(X_train_grades, y_train_grades)

# Make predictions for the test set
y_pred_grades = pipeline.predict(X_test_grades)

# Evaluate the model
mse_grades = mean_squared_error(y_test_grades, y_pred_grades)
r2_grades = r2_score(y_test_grades, y_pred_grades)

# Streamlit section
st.title("ScoreVision: Grade Prediction System")

# Show evaluation metrics
st.subheader("Model Evaluation")
st.write(f"Mean Squared Error for predicting grades: {mse_grades}")
st.write(f"R-squared value for predicting grades: {r2_grades:.2f}")

# Input fields for the user to predict grades
st.subheader("Input Student Data for Grade Prediction")
hours_studied = st.number_input("Hours Studied", min_value=0.0, value=0.0)
attendance = st.number_input("Attendance", min_value=0.0, value=0.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, value=0.0)
previous_scores = st.number_input("Previous Scores", min_value=0.0, value=0.0)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0.0, value=0.0)
physical_activity = st.number_input("Physical Activity", min_value=0.0, value=0.0)

# Categorical inputs
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.selectbox("School Type", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Dictionary for the input data
student_data = {
    'Hours_Studied': [hours_studied],
    'Attendance': [attendance],
    'Sleep_Hours': [sleep_hours],
    'Previous_Scores': [previous_scores],
    'Tutoring_Sessions': [tutoring_sessions],
    'Physical_Activity': [physical_activity],
    
    # Categorical variables
    'Parental_Involvement': [parental_involvement],
    'Access_to_Resources': [access_to_resources],
    'Extracurricular_Activities': [extracurricular_activities],
    'Motivation_Level': [motivation_level],
    'Internet_Access': [internet_access],
    'Family_Income': [family_income],
    'Teacher_Quality': [teacher_quality],
    'School_Type': [school_type],
    'Peer_Influence': [peer_influence],
    'Learning_Disabilities': [learning_disabilities],
    'Parental_Education_Level': [parental_education_level],
    'Distance_from_Home': [distance_from_home],
    'Gender': [gender]
}

# Convert to DataFrame
student_data_df = pd.DataFrame(student_data)

# Prediction
if st.button("Predict Grade"):
    predicted_grade = pipeline.predict(student_data_df)
    st.success(f"The predicted grade (Exam Score) is: {predicted_grade[0]:.2f}")

# Show the scatter plot of predicted vs actual grades
if st.button("Show Predicted vs Actual"):
    fig, ax = plt.subplots()
    ax.scatter(y_test_grades, y_pred_grades, color="blue", label="Predicted vs Actual")
    ax.plot([min(y_test_grades), max(y_test_grades)], [min(y_test_grades), max(y_test_grades)], color="red", label="Perfect Fit")
    ax.set_xlabel("Actual Grades")
    ax.set_ylabel("Predicted Grades")
    ax.set_title("Predicted vs Actual Grades")
    ax.legend()
    st.pyplot(fig)
