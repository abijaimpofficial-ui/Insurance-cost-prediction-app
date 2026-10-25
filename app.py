"""
Streamlit interface for Insurance Cost Prediction.
Run:  streamlit run app.py      (after running insurance_model.py once)
"""
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰")
st.title("💰 Medical Insurance Cost Predictor")
st.caption("Ridge / Lasso Regression model")

MODEL_PATH = "insurance_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python insurance_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 64, 35)
    bmi = st.number_input("BMI", 15.0, 55.0, 27.0, step=0.1)
    children = st.selectbox("Number of children", [0, 1, 2, 3, 4, 5])
with col2:
    sex = st.radio("Sex", ["male", "female"], horizontal=True)
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Cost", type="primary"):
    row = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region,
    }])
    # same engineered features as in training
    row["smoker_bmi"] = (row["smoker"] == "yes").astype(int) * row["bmi"]
    row["age_sq"] = row["age"] ** 2
    pred = max(float(model.predict(row)[0]), 0)
    st.success(f"Estimated yearly insurance charge: **${pred:,.2f}**")
    st.caption("Estimate from a statistical model - not a real insurance quote.")
