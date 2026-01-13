import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Travel Prediction", layout="centered")

st.title("✈️ Sayohat yo‘nalishini bashorat qilish")
st.write("Ichki yoki Xorijiy sayohatni aniqlash")

# Modelni yuklash
model = joblib.load("model.pkl")

# Inputlar
age = st.number_input("Yosh", min_value=1, max_value=100, value=30)
duration = st.number_input("Davomiyligi (kun)", min_value=1, max_value=60, value=7)
cost = st.number_input("Umumiy xarajat", min_value=100, max_value=10000, value=1000)

if st.button("Natijani ko‘rish"):
    df = pd.DataFrame([[age, duration, cost]],
                      columns=["Traveler age", "Duration (days)", "Accommodation cost"])

    prediction = model.predict(df)

    if prediction[0] == 0:
        st.success("🟢 Ichki sayohat")
    else:
        st.success("🌍 Xorijiy sayohat")
