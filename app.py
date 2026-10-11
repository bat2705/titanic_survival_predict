import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_pipeline():
    return joblib.load(os.path.join(BASE_DIR, "titanic_pipeline.pkl"))


pipeline = load_pipeline()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict the likelihood of survival.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

if st.button("Predict Survival"):
    try:
        passenger = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
        }])

        probability = float(pipeline.predict_proba(passenger)[0][1])

        if probability >= 0.5:
            st.success("🎉 The model predicts this passenger would survive.")
        else:
            st.error("The model predicts this passenger would not survive.")

        st.write(f"Predicted survival probability: {probability:.1%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")