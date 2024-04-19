import streamlit as st
import numpy as np
import pickle

jobs_dict = {0: 'AI ML Specialist', 1: 'API Integration Specialist', 2: 'Application Support Engineer',
             3: 'Business Analyst', 4: 'Customer Service Executive', 5: 'Cyber Security Specialist',
             6: 'Data Scientist', 7: 'Database Administrator', 8: 'Graphics Designer', 9: 'Hardware Engineer',
             10: 'Helpdesk Engineer', 11: 'Information Security Specialist', 12: 'Networking Engineer',
             13: 'Project Manager', 14: 'Software Developer', 15: 'Software Tester', 16: 'Technical Writer'}

# Load the trained model
loaded_model = pickle.load(open("technical.pkl", "rb"))

# Define the skills names
skills_names = [
    "Database Fundamentals",
    "Computer Architecture",
    "Distributed Computing Systems",
    "Cyber Security",
    "Networking",
    "Development",
    "Programming Skills",
    "Project Management",
    "Computer Forensics Fundamentals",
    "Technical Communication",
    "AI ML",
    "Software Engineering",
    "Business Analysis",
    "Communication skills",
    "Data Science",
    "Troubleshooting skills",
    "Graphics Designing",
]

# Create a Streamlit UI
st.title("Career Prediction App")

# Input for user to enter skill levels
st.sidebar.header("Enter Skill Levels")
user_skills = []
for i, skill_name in enumerate(skills_names):
    user_input = st.sidebar.slider(f"{skill_name} (0-10)", 0, 10, 5)
    user_skills.append(user_input)

# Button to make predictions
if st.sidebar.button("Predict Career"):
    # Convert user input to NumPy array
    input_data = np.array(user_skills).reshape(1, -1)

    # Make predictions
    predictions = loaded_model.predict(input_data)
    probabilities = loaded_model.predict_proba(input_data)

    # Display predictions
    # Display predictions
    # Display predictions
    st.subheader("Predicted Career:")
    predicted_career = jobs_dict.get(predictions[0], "Unknown Career")
    print(f"predictions[0]: {predictions[0]}")
    print(f"jobs_dict keys: {list(jobs_dict.keys())}")
    print(f"jobs_dict values: {list(jobs_dict.values())}")
    st.write(predicted_career)

    # Display probabilities
    st.subheader("Prediction Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        st.write(f"{jobs_dict[i]}: {prob * 100:.2f}%")

    # Display additional information based on probability threshold
    threshold = 5  # You can adjust this threshold
    selected_jobs = [
        jobs_dict[i]
        for i, prob in enumerate(probabilities[0])
        if prob > threshold / 100
    ]
    if selected_jobs:
        st.subheader("Potential Careers (Probability > 5%):")
        st.write(selected_jobs)
    else:
        st.subheader("No potential careers found.")

# Add some information about the skills
st.sidebar.header("About the Skills")
st.sidebar.info(
    "Use the sliders to input your skill levels (0-10). 0 means no knowledge, and 10 means expert level."
)

# Add information about the app
st.sidebar.header("About the App")
st.sidebar.info(
    "This app uses a trained KNeighborsClassifier to predict potential careers based on input skill levels."
)

# # Acknowledgment


# # Display a link to the source code on GitHub
# st.sidebar.markdown("[View Source Code](https://github.com/your-username/your-repo)")
