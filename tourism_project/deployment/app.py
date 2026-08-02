"""Streamlit front-end for the Wellness Tourism Package purchase predictor.

Streamlit Community Cloud runs this file directly from the repo. It loads the
model committed by the pipeline, collects user inputs, and shows a prediction.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_tourism_model.joblib"

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="✈️")
st.title("Wellness Tourism Package — Purchase Predictor")
st.write(
    "Enter a customer's details to predict whether they are likely to "
    "purchase the Wellness Tourism Package."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --- Collect inputs ---
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch (min)", min_value=0.0, max_value=120.0, value=15.0)
    Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    NumberOfFollowups = st.number_input("Number of Followups", min_value=0.0, max_value=10.0, value=3.0)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
with col2:
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfTrips = st.number_input("Number of Trips (per year)", min_value=0.0, max_value=50.0, value=2.0)
    Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    OwnCar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=10.0, value=0.0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=20000.0)

if st.button("Predict"):
    row = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    if pred == 1:
        st.success(f"Likely to PURCHASE the package (probability {proba:.1%}).")
    else:
        st.info(f"Unlikely to purchase the package (probability {proba:.1%}).")
