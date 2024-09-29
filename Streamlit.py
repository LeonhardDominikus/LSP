import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import streamlit as st

# Judul aplikasi
st.title("Analisis Bulan Dengan Intensita Hujan Tertinggi")

# Membaca dataset secara langsung
df = pd.read_csv('JumlahHariHujan2022.csv')  # Pastikan file CSV ada di direktori yang sama
st.write("Dataset:")
st.dataframe(df.head(20))  # Menampilkan DataFrame

# Menampilkan info dan deskripsi dataset
st.write("Info Dataset:")
buffer = pd.DataFrame(df.info())

st.write("Deskripsi Dataset:")
st.write(df.describe())

st.write("Jumlah Nilai Null pada Setiap Kolom:")
st.write(df.isnull().sum())

# Mengubah kolom 'Bulan' menjadi numerik dengan LabelEncoder
le = LabelEncoder()
df['Bulan_numeric'] = le.fit_transform(df['Bulan'])

# Variabel input (X) adalah semua kolom stasiun
X = df[['Stasiun Meteorologi Jatiwangi', 'Stasiun Meteorologi Citeko', 'Stasiun Klimatologi Bogor', 'Stasiun Geofisika Bandung']]
# Variabel target (y) adalah 'Bulan_numeric'
y = df['Bulan_numeric']

# Split data menjadi data latih dan uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # Latih model dengan data latih
lr_pred = lr_model.predict(X_test)  # Prediksi dengan data uji

# Membulatkan hasil prediksi dan membatasi rentangnya
lr_pred_rounded = np.clip(np.round(lr_pred), 0, len(le.classes_) - 1)

# Evaluasi model Linear Regression dengan Mean Squared Error
mse = mean_squared_error(y_test, lr_pred)
st.write("Mean Squared Error (Linear Regression):", mse)

# Plot hasil prediksi Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Linear Regression: Prediksi Bulan sebagai Angka')
plt.xlabel('Bulan Aktual')
plt.ylabel('Prediksi Bulan')
st.pyplot(plt)

# Mengembalikan prediksi bulan ke bentuk aslinya
predicted_months_lr = le.inverse_transform(lr_pred_rounded.astype(int))
actual_months_lr = le.inverse_transform(y_test)

st.write("Bulan Prediksi (Linear Regression):", predicted_months_lr)
st.write("Bulan Aktual (Linear Regression):", actual_months_lr)

# Logistic Regression
log_reg_model = LogisticRegression(max_iter=200)
log_reg_model.fit(X_train, y_train)  # Latih model dengan data latih
log_reg_pred = log_reg_model.predict(X_test)  # Prediksi dengan data uji

# Hitung akurasi
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
st.write('Akurasi Logistic Regression: ', log_reg_accuracy * 100, '%')

# Visualisasi hasil prediksi Logistic Regression
plt.figure(figsize=(10, 6))
conf_matrix_lr = confusion_matrix(y_test, log_reg_pred)
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix Logistic Regression Prediksi Bulan")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(plt)

# Mengembalikan prediksi bulan ke bentuk aslinya
predicted_months_log = le.inverse_transform(log_reg_pred)
actual_months_log = le.inverse_transform(y_test)

st.write("Bulan Prediksi (Logistic Regression):", predicted_months_log)
st.write("Bulan Aktual (Logistic Regression):", actual_months_log)

# Standarisasi fitur untuk K-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Standarisasi data latih
X_test_scaled = scaler.transform(X_test)  # Standarisasi data uji

# K-Nearest Neighbors (K-NN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)  # Latih model dengan data latih
knn_pred = knn_model.predict(X_test_scaled)  # Prediksi dengan data uji

# Hitung akurasi
knn_accuracy = accuracy_score(y_test, knn_pred)
st.write('Akurasi K-NN: ', knn_accuracy * 100, '%')

# Visualisasi hasil prediksi K-NN
plt.figure(figsize=(10, 6))
conf_matrix_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix K-NN Prediksi Bulan")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(plt)

plt.clf()

# Mengembalikan prediksi bulan ke bentuk aslinya
predicted_months_knn = le.inverse_transform(knn_pred)
actual_months_knn = le.inverse_transform(y_test)

st.write("Bulan Prediksi (K-NN):", predicted_months_knn)
st.write("Bulan Aktual (K-NN):", actual_months_knn)

st.write("Dengan menggunakan ke3 model disimpulkan bahwa, model akurasi tertinggi adalah Linear Regression dengan prediksi 33%")