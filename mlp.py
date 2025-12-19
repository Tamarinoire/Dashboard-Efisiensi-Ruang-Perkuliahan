
# K-MEANS + MLP TRAINING PIPELINE


import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras.models import Sequential
from keras.layers import Dense, Dropout




# K-MEANS CLUSTERING


print("🔹 Load data untuk K-Means...")
df = pd.read_csv(
    r"C:\Users\ASUS\OneDrive\Desktop\Streamlit\room-efficiency-app\data\deep learning - percobaan.csv"
)


features = [
    "Luas Ruangan",
    "Waktu penggunaan",
    "Rasio Terpakai",
    "Jumlah AC"
]

X = df[features].copy()

scaler_kmeans = StandardScaler()
X_scaled = scaler_kmeans.fit_transform(X)

print("🔹 Evaluasi jumlah cluster...")
k_range = range(2, 6)
inertias = []
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k = 3
print(f"🔹 Jumlah cluster dipilih: {best_k}")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans_final.fit_predict(X_scaled)

df.to_excel("clustering_results.xlsx", index=False)
print("✅ Hasil clustering disimpan ke clustering_results.xlsx")


# MULTI-LAYER PERCEPTRON (MLP)


print("🔹 Load data untuk MLP...")
dataset = pd.read_excel("clustering_results.xlsx")

X = dataset[features].values
y = dataset["Cluster"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

print("🔹 Bangun model MLP...")
model = Sequential()
model.add(Dense(32, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(3, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🔹 Training MLP...")
model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)

print("🔹 Evaluasi model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Akurasi data uji: {accuracy * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# SIMPAN MODEL & SCALER


os.makedirs("model", exist_ok=True)

model.save("model/mlp_model.h5")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model MLP dan scaler berhasil disimpan!")
