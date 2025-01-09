import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.preprocessing import StandardScaler

# KNN Manual
class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict(self, x):
        # Hitung jarak ke semua sampel pelatihan
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # Ambil k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k]
        k_neighbors = [self.y_train[i] for i in k_indices]
        # Tentukan kelas berdasarkan mayoritas
        most_common = Counter(k_neighbors).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return [self._predict(x) for x in np.array(X)]

# Fungsi untuk normalisasi data (Standardization)
def normalize(X):
    # Normalisasi dengan rumus standar (mean=0, std=1)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Fungsi untuk membagi data menjadi training dan testing secara manual
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# Fungsi untuk menghitung Confusion Matrix secara manual
def generate_confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for true, pred in zip(y_true, y_pred):
        true_index = np.where(labels == true)[0][0]
        pred_index = np.where(labels == pred)[0][0]
        cm[true_index, pred_index] += 1

    return cm, labels

# Fungsi untuk menghitung akurasi secara manual
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Fungsi untuk menyimpan model ke file joblib
def save_model(model, filename):
    with open(filename, 'wb') as file:
        joblib.dump(model, file)

# Fungsi untuk memuat model dari file joblib
def load_model(filename):
    with open(filename, 'rb') as file:
        return joblib.load(file)

# 1. Dataset: Pastikan dataset tersedia dalam variabel df
df=pd.read_csv('dataset_fiks.csv')
# Pastikan df adalah DataFrame yang memiliki kolom fitur dan target.

# 2. Menggabungkan kolom teks menjadi satu kolom untuk analisis TF-IDF
df['combined_text'] = df['Interests'] + ' ' + df['Skills'] + ' ' + df['Certification Course Title'] + ' ' + df['UG Specialization (Major)']

# 3. Menggunakan TF-IDF untuk mengubah teks menjadi fitur
vectorizer = TfidfVectorizer(max_features=1000)  # Set max_features sesuai kebutuhan
tfidf_features = vectorizer.fit_transform(df['combined_text']).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=vectorizer.get_feature_names_out())

# 4. Menggabungkan fitur TF-IDF dengan fitur numerik lainnya
df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# 5. Menghapus kolom teks asli dan kolom sementara
df = df.drop(columns=['Interests', 'Skills', 'Certification Course Title', 'UG Specialization (Major)', 'combined_text'])

# 6. Pisahkan Fitur dan Target
X = df.drop(columns=['Mapped Category'])  # Fitur
y = df['Mapped Category']  # Target

# Ubah kolom kategorikal menjadi numerik
for col in X.columns:
    if X[col].dtype == 'object':  # Jika kolom bertipe string atau object
        print(f"Encoding column: {col}")
        X[col] = X[col].astype('category').cat.codes

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
X_normalized = scaler.fit_transform(X)

# Save the scaler parameters
save_model(scaler, 'normalization_params.joblib')


# 8. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y.values, test_size=0.2, random_state=42)

# 9. Membuat dan Melatih Model KNN
knn_model = KNN(k=17)  
knn_model.fit(X_train, y_train)

# 10. Evaluasi Model
y_pred = knn_model.predict(X_test)

# 11. Menghitung dan Menampilkan Akurasi
accuracy_value = accuracy(y_test, y_pred)

# 12. Menyimpan Model ke File
save_model(knn_model, "knn_model.joblib")

# 13. Memuat Model dan Menguji Ulang
loaded_knn_model = load_model("knn_model.joblib")
loaded_y_pred = loaded_knn_model.predict(X_test)
loaded_accuracy_value = accuracy(y_test, loaded_y_pred)


# Simpan TF-IDF Vectorizer ke file
joblib.dump(vectorizer, 'vectorizer.joblib')



