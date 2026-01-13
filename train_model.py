import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 1. Datasetni yuklash
df = pd.read_csv("travel.csv")

# 2. Ichki / Xorijiy belgisi
internal_countries = ['Tashkent', 'Samarkand', 'Bukhara', 'Khiva', 'Fergana']
df['TravelType'] = df['Destination'].apply(
    lambda x: 'Ichki' if isinstance(x, str) and any(c in x for c in internal_countries) else 'Xorijiy'
)

# 3. Feature va Target
X = df.drop(['TravelType'], axis=1)
y = df['TravelType']

# 4. NaN qiymatlarni to‘ldirish
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='object').columns

X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])
X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

# 5. Label Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# 6. Scaling (KNN uchun MUHIM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 8. KNN modeli
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 9. Baholash
y_pred = knn.predict(X_test)
print("CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

# 10. Modelni saqlash
joblib.dump(knn, "model/knn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

print("\n✅ Model va fayllar muvaffaqiyatli saqlandi!")
