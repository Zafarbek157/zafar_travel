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
# 2. Foydalanuvchi inputlari
# ----------------------
age = st.number_input("Yosh", min_value=1, max_value=100, value=30)
duration = st.number_input("Davomiyligi (kun)", min_value=1, max_value=60, value=7)
cost = st.number_input("Umumiy xarajat", min_value=100, max_value=10000, value=1000)

# Agar boshqa kategorik ustunlar train_model’da bo‘lsa, default qiymat beramiz
# Misol: gender, accommodation_type va h.k.
default_cats = {col: 0 for col in label_encoders.keys()}  # 0 = train’dagi default label

if st.button("Natijani ko‘rish"):

    # ----------------------
    # 3. Inputni DataFrame formatiga keltirish
    # ----------------------
    df_input = pd.DataFrame([[age, duration, cost]],
                            columns=["Traveler age", "Duration (days)", "Accommodation cost"])

    # Boshqa kategorik ustunlar qo‘shish
    for col, default_val in default_cats.items():
        if col not in df_input.columns:
            df_input[col] = default_val

    # ----------------------
    # 4. LabelEncoder bilan kodlash
    # ----------------------
    for col, le in label_encoders.items():
        val = df_input[col].astype(str).values[0]
        if val in le.classes_:
            df_input[col] = le.transform([val])
        else:
            # Yangi label kelganda default 0 ishlatamiz
            df_input[col] = 0

    # ----------------------
    # 5. Scaling
    # ----------------------
    df_scaled = scaler.transform(df_input[feature_columns])

    # ----------------------
    # 6. Prediction
    # ----------------------
    prediction = model.predict(df_scaled)
    
    if target_encoder:
        pred_label = target_encoder.inverse_transform(prediction)[0]
    else:
        pred_label = "Ichki" if prediction[0] == 0 else "Xorijiy"

    # ----------------------
    # 7. Natijani chiqarish
    # ----------------------
    if pred_label == "Ichki":
        st.success("🟢 Ichki sayohat")
    else:
        st.success("🌍 Xorijiy sayohat")
