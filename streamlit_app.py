import streamlit as st
import pandas as pd
import joblib

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
# 2. Inputlar
# ----------------------
age = st.number_input("Yosh", min_value=1, max_value=100, value=30)
duration = st.number_input("Davomiyligi (kun)", min_value=1, max_value=60, value=7)
cost = st.number_input("Umumiy xarajat", min_value=100, max_value=10000, value=1000)

# Agar boshqa kategorik ustunlar bo‘lsa, ularga default qiymat berish
# Misol uchun, train_model’da "Accommodation Type" yoki "Traveler Gender" bo‘lsa:
# gender = "male"  # default
# accommodation = "hotel"

if st.button("Natijani ko‘rish"):

    # ----------------------
    # 3. Inputni DataFrame formatiga keltirish
    # ----------------------
    df_input = pd.DataFrame([[age, duration, cost]],
                            columns=["Traveler age", "Duration (days)", "Accommodation cost"])

    # Agar train_model’dagi barcha ustunlar kerak bo‘lsa, default qiymatlar qo‘shish
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0  # numeric default, kategorik bo‘lsa, 0 (label encoded) ham ishlaydi

    # ----------------------
    # 4. LabelEncoder bilan kodlash (faqat kerak bo‘lsa)
    # ----------------------
    for col, le in label_encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col].astype(str))

    # ----------------------
    # 5. Scaling
    # ----------------------
    df_scaled = scaler.transform(df_input[feature_columns])

    # ----------------------
    # 6. Prediction
    # ----------------------
    prediction = model.predict(df_scaled)
    pred_label = target_encoder.inverse_transform(prediction) if target_encoder else prediction

    # ----------------------
    # 7. Natijani chiqarish
    # ----------------------
    if pred_label[0] == "Ichki" or pred_label[0] == 0:
        st.success("🟢 Ichki sayohat")
    else:
        st.success("🌍 Xorijiy sayohat")
