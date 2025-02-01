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
data = pd.read_csv("StudentPerformanceFactors.csv")

# Define the feature and target variable for predicting the grades
X_grades = data.drop(columns=['Exam_Score'])
y_grades = data['Exam_Score']

# Identify categorical columns
categorical_cols = X_grades.select_dtypes(include=['object']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[('num', 'passthrough', X_grades.select_dtypes(exclude=['object']).columns),
                  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])

# Split the dataset into training and testing sets for predicting grades
X_train_grades, X_test_grades, y_train_grades, y_test_grades = train_test_split(X_grades, y_grades, test_size=0.1, random_state=5981)

# Train the linear regression model
pipeline.fit(X_train_grades, y_train_grades)

# Make predictions for the test set
y_pred_grades = pipeline.predict(X_test_grades)

# Evaluate the model
mse_grades = mean_squared_error(y_test_grades, y_pred_grades)
r2_grades = r2_score(y_test_grades, y_pred_grades)
st.write(f"Mean Squared Error for predicting grades: {mse_grades}")
st.write(f"R-squared value for predicting grades: {r2_grades:.2f}")

# Create or connect to SQLite database for login/signup
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

# Function for signing up a new user
def sign_up(username, password):
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        return "Username already exists"
    else:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return "Sign-up successful!"

# Function for logging in an existing user
def login(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    if cursor.fetchone():
        return True
    else:
        return False

# Function to plot the graph
def plot_predicted_vs_actual():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test_grades, y_pred_grades, color="blue", label="Predicted vs Actual")
    ax.plot([min(y_test_grades), max(y_test_grades)], [min(y_test_grades), max(y_test_grades)], color="red", label="Perfect Fit")

    ax.set_xlabel("Actual Grades")
    ax.set_ylabel("Predicted Grades")
    ax.set_title("Predicted vs Actual Grades")
    ax.legend()
    
    st.pyplot(fig)

# Function to predict grades based on the user input
def predict_grade(inputs):
    try:
        student_data = pd.DataFrame(inputs)
        # Ensure categorical variables have correct data types (string/object)
        for col in categorical_cols:
            student_data[col] = student_data[col].astype(str)

        # Predict the grade
        predicted_grade = pipeline.predict(student_data)

        # Show the prediction result
        st.write(f"The predicted grade (Exam Score) is: {predicted_grade[0]:.2f}")

    except ValueError as e:
        st.warning(f"Please enter valid values for all fields. Error: {e}")

# Main Streamlit app
st.title("Student Grade Prediction App")

# User authentication (Login or Sign Up)
auth_option = st.selectbox("Select an option", ["Login", "Sign Up"])

if auth_option == "Login":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.success(f"Welcome, {username}!")
            
            # Collect user input for grade prediction
            hours_studied = st.number_input("Hours Studied", min_value=0.0)
            attendance = st.number_input("Attendance", min_value=0.0)
            parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
            access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
            extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
            previous_scores = st.number_input("Previous Scores", min_value=0.0)
            motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])
            tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0.0)
            family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
            teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
            school_type = st.selectbox("School Type", ["Public", "Private"])
            peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
            physical_activity = st.number_input("Physical Activity", min_value=0.0)
            learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
            parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
            distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
            gender = st.selectbox("Gender", ["Male", "Female"])

            user_inputs = {
                'Hours_Studied': [hours_studied],
                'Attendance': [attendance],
                'Sleep_Hours': [sleep_hours],
                'Previous_Scores': [previous_scores],
                'Tutoring_Sessions': [tutoring_sessions],
                'Physical_Activity': [physical_activity],
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
            
            predict_button = st.button("Predict Grade")
            if predict_button:
                predict_grade(user_inputs)

            plot_button = st.button("Show Predicted vs Actual")
            if plot_button:
                plot_predicted_vs_actual()

        else:
            st.error("Invalid username or password")

elif auth_option == "Sign Up":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        message = sign_up(username, password)
        st.write(message)
