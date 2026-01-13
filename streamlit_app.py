import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Travel Prediction", layout="centered")

st.title("✈️ Sayohat yo‘nalishini bashorat qilish")
st.write("Ichki yoki Xorijiy sayohatni aniqlash")

# ----------------------
# 1. Model va preprocessing fayllarini yuklash
# ----------------------
model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
target_encoder = joblib.load("model/target_encoder.pkl") if "target_encoder.pkl" in st.session_state else None

# ----------------------
# 2. Session state bilan tugma
# ----------------------
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

def run_prediction():
    st.session_state.run_prediction = True

# ----------------------
# 3. Foydalanuvchi inputlari
# ----------------------
age = st.number_input("Yosh", min_value=1, max_value=100, value=30)
duration = st.number_input("Davomiyligi (kun)", min_value=1, max_value=60, value=7)
cost = st.number_input("Umumiy xarajat", min_value=100, max_value=10000, value=1000)

st.button("Natijani ko‘rish", on_click=run_prediction)

# ----------------------
# 4. Prediction faqat tugma bosilganda
# ----------------------
if st.session_state.run_prediction:

    # DataFrame yaratish
    df_input = pd.DataFrame([[age, duration, cost]],
                            columns=["Traveler age", "Duration (days)", "Accommodation cost"])

    # Feature columns dagi barcha ustunlar mavjudligini tekshirish va default qiymat berish
    for col in feature_columns:
        if col not in df_input.columns:
            if col in label_encoders:
                df_input[col] = 0  # kategorik default label 0
            else:
                df_input[col] = 0  # numeric default 0

    # LabelEncoder bilan kodlash
    for col, le in label_encoders.items():
        if col in df_input.columns:
            val = df_input[col].astype(str).values[0]
            if val in le.classes_:
                df_input[col] = le.transform([val])
            else:
                df_input[col] = 0  # yangi label uchun default 0

    # Scaling
    df_scaled = scaler.transform(df_input[feature_columns])

    # Prediction
    prediction = model.predict(df_scaled)
    if target_encoder:
        pred_label = target_encoder.inverse_transform(prediction)[0]
    else:
        pred_label = "Ichki" if prediction[0] == 0 else "Xorijiy"

    # Natijani chiqarish
    if pred_label == "Ichki":
        st.success("🟢 Ichki sayohat")
    else:
        st.success("🌍 Xorijiy sayohat")
