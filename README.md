# Wine Quality Analysis
# Konteks
Dalam industri wine, kualitas produk menjadi salah satu faktor penting yang memengaruhi kepuasan konsumen, reputasi produsen, dan nilai jual produk di pasaran. Kualitas wine dipengaruhi oleh berbagai karakteristik kimia seperti tingkat keasaman _(acidity)_, kadar alkohol, pH, kandungan sulfur dioxide, chlorides, sulphates, dan beberapa komponen lainnya yang terkandung dalam wine.

Penilaian kualitas wine umumnya dilakukan oleh para ahli melalui proses _wine testing_. Namun proses tersebut membutuhkan banyak waktu, biaya, dan dapat dipengaruhi oleh subjektivitas penilai. Oleh karena itu, diperlukan suatu pendekatan yang mampu membantu proses penilaian kualitas wine secara lebih cepat, efisien, dan konsisten.

Melalui perkembangan teknologi di bidang _Data Mining_ dan _Machine Learning_, kualitas wine dapat diprediksi berdasarkan pola dari data karakteristik kimia yang dimiliki setiap sampel wine. Dengan memanfaatkan algoritma klasifikasi, model _machine learning_ dapat mempelajari hubungan antara karakteristik kimia wine dengan kualitasnya sehingga mampu memberikan prediksi kualitas secara otomatis pada data naru.

# Problem Statement
Permasalahan dalam proyek ini adalah bagaimana membuat model klasifikasi yang mampu memprediksi nilai `quality` pada wine berdasarkan fitur-fitur kimia yang tersedia. Dataset training memiliki kolom `quality` sebagai target, sedangkan dataset testing tidak memiliki kolom `quality`, sehingga nilai tersebut perlu diprediksi menggunakan model machine learning yang telah dibuat.

Dalam proyek ini dilakukan pembuatan model _machine learning_ untuk mengklasifikasikan kualitas wine menggunakan beberapa model klasifikasi seperti _Logistic Regression_, KNN, SVM _(Support Vector Machine)_, _Random Forest_, dan _Decision Tree_. Selanjutnya akan dilakukan evaluasi dan perbandingan performa model untuk menentukan model terbaik dalam memprediksi kualitas wine. 

# Tujuan
Proyek ini betujuan untuk membuat model klasifikasi yang dapat digunakan dalam memprediksi kualitas wine pada dataset testing. Model tersebut dibuat menggunakan dataset training yang  telah memiliki kolom `quality` sebagai target prediksi.

# Dataset
Dataset wine quality memiliki beberapa variabel yaitu:
| Variabel | Deskripsi |
|---|---|
| `fixed acidity` | Tingkat keasaman tetap pada wine |
| `volatile acidity` | Tingkat keasaman yang mudah menguap |
| `citric acid` | Kandungan asam sitrat pada wine |
| `residual sugar` | Jumlah gula yang tersisa setelah fermentasi |
| `chlorides` | Kandungan garam pada wine |
| `free sulfur dioxide` | Jumlah sulfur dioksida bebas |
| `total sulfur dioxide` | Total sulfur dioksida dalam wine |
| `density` | Massa jenis wine |
| `pH` | Tingkat keasaman wine |
| `sulphates` | Kandungan sulfat pada wine |
| `alcohol` | Kadar alkohol pada wine |
| `quality` | Nilai kualitas wine |
| `Id` | Identitas data |

# 1. Persiapan Data
# Import Library
Pada tahap awal analisis, dilakukan proses import beberapa library Python yang digunakan untuk membantu proses pengolahan data, visualisasi, preprocessing, pembuatan model _machine learning_, serta evaluasi model.

Library `pandas` dan `numpy` digunakan untuk membaca serta mengolah dataset. Library `matplotlib.pyplot` dan `seaborn` digunakan untuk membuat visualisasi data agar pola dan hubungan antar variabel lebih mudah dipahami.

Selain itu, di sini juga menggunakan beberapa modul dari `scikit-learn` seperti `train_test_split` untuk membagi data training dan testing, `LabelEncoder` dan `StandardScaler` untuk prerocessing data, serta beberapa model klasifikasi seperti `DecisionTreeClassifier`, `KneighborsClassifier`, `GaussianNB` untuk membuat model _machine learning_.

Untuk tahap evaluasi model, digunakan metrik seperti `accuracy_score`, `classification_report`, dan `confusion_matrix` untuk mengetahui performa model dalam melakukan klasifikasi.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
# Membaca Dataset
Pada tahap ini dilakukan proses membaca dataset training menggunakan library `pandas`. Fungsi `pd.read_csv()` digunakan untuk membaca file dengan format CSV dan mengubahkan menjadi bentuk DataFrame sehingga data dapat dianalisis lebih lanjut. Dataset training disimpan ke dalam variabel `data_training` yang berisi seluruh data fitur serta target `quality` yang akan digunakan dalam proses pelatihan model _machine learning_. Sedangkan dataset testing disimpan pada variabel `data_testing` yang berisi fitur-fitur kimia wine tanpa kolom `quality` dan akan digunakan untuk proses prediksi.

Selanjutnya kedua variabel dipanggil untuk menampilkan isi dataset sehingga dapat diketahui struktur data, jumlah kolom, serta bebera data awal yang terdapat pada dataset.
```python
data_training = pd.read_csv('data_training.csv')
data_training
```
```python
data_testing = pd.read_csv('data_testing.csv')
data_testing
```

# 2. Pembersihan Data
# Informasi Dataset
Pada tahap ini digunakan  `info()` untuk menampilkan informasi umum mengenai dataset training. Informasi yang ditampilkan meliputi jumlah baris dan kolom, nama kolom, tipe data pada setiap kolom, serta jumlah data yang tidak kosong _(non-null)_. Melalui informasi ini, dapat diketahui apakah terdapat _missing values_ pada dataset serta memastika bahwa tipe data pada setiap variabel sudah sesuai untuk proses analisis dan pembuatan model _machine learning_.
```python
print(data_training.info())
```
# Statistik Deskriptif dan Pengecekan Missing Values
Pada tahap ini dilakukan analisis awal terhadap dataset menggunakan fungsi `describe()` dan `isnull().sum()`. Fungsi `describe()` digunakan untuk menampilkan statistik deskriptif dari setiap variabel numerik pada dataset seperti nilai rata-rata _(mean)_, standar deviasi _(std)_, nilai minimum, nilai maksimum, serta kuartil data. Informasi ini membantu dalam memahami distribusi dan karakteristik data sebelum dilakukan proses _modelling_. Sedangkan fungsi `isnull().sum()` digunakan untuk mengecek jumlah _missing values_ pada setiap kolom dataset. Proses ini harus dilakukan untuk memastikan tidak terdapat data kosong yang dapat memengaruhi performa model _machine learning_.
```python
print(data_training.describe())
print(data_training.isnull().sum())
```
Berdasarkan hasil output, diperoleh hasil dataset training memiliki 857 data pada setiap variabel yang menunjukkan bahwa seluruh kolom memiliki jumlah data yang sama. Nilai statistik seperti mean, minimum, maximum, dan quartile memberikan gambar mengenai distribusi data pada setiap fitur. Misalnya variabel `alkohol` memiliki rata-rata sebesar 10.43 dengan nilai minimum 8.4 dan maksimum 14.0 yang menunjukkan variasi kadar alkohol pada wine. 

Selain itu, hasil dari `isnull().sum()` menunjukkan bahwa seluruh kolom memiliki nilai 0 yang berarti tidak terdapat _missing values_ pada dataset tarining. Sehingga, data sudah lengkap dan dapat langsung digunakan pada tahap preprocessing dan pembuatan model _machine learning_. 

# 3. Pembuatan Model
# Memisahkan Fitur dan Target
Pada tahap ini dilakukan pemisahan antara fitur _(feature)_ dan target _(label)_ pada dataset training. Pemisahan ini dilakukan agar model _machine learning_ dapat mempelajari hubungan antara variabel input dengan taregt yang akan diprediksi. Variaebl `x` digunakan untuk menyimpan seluruh fitur prediktor dengan menghapus kolom `quality` dan `Id`. Kolom `quality` dihapus karena merupakan target prediksi, sedangkan kolom `Id` tidak digunakan dalam proses _modelling_ karena hanya berfungsi sebagai identitas data. Selanjutnya variabel `y` digunakan untuk menyimpan kolom `quality` sebagai taregt klasifikasi yang akan dipelajari oleh model _machine learning_. 

Pada dataset testing, dilakukan penghapusan kolom `Id` dan hasilnya disimpan pada variabel `x_testing_final`. Dataset testing tidak memiliki kolom `quality` karena nilai tersebut akan diprediksi menggunakan model yang telah dilatih sebelumnya.
```python
x = data_training.drop(['quality','Id'], axis=1)
y = data_training['quality']

x_test_final=data_testing.drop(['Id'], axis=1)
```

# Pembagian Data Training dan Validation
Pada tahap ini dilakukan proses pembagian dataset menjadi data training dan data validation menggunakan `train_test_split()` dari `scikit-learn`. Pembagian data bertujuan agar model dapat dilatih menggunakan sebagian data _(training data)_ dan diuji menggunakan data lainnya _(validation data)_ untuk mengetahui performa model terhadap data yang belum pernah dilihat sebelumnya.

Variabel `x_train` dan `y_train` digunakan sebagai data pelatihan model, sedangkan `x_val` dan `y_val` digunakan sebagai data validasi untuk evaluasi model. Parameter `test_size=0.2` menunjukkan bahwa sebesar 20% data digunakan sebagai data validation, sedangkan 80% sisanya digunakan sebagai data training. Parameter `random_state=42` digunakan agar hasil pembagian data tetap konsisten setiap kali program dijalankan. Sedangkan parameter `stratify=y` digunakan untuk menjaga proporsi setiap kelas `quality` pada data training dan validation tetap seimbang sehingga distribusi data tidak berubah secara signifikan.
```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
```

# Feature Scalling
Pada tahap ini dilakukan proses _feature scalling_ menggunakan `StandardScaler` dari `scikit-learn`. Feature scalling betujuan untuk menyamakan rentang nilai antar fitur agar tidak terdapat fitur yang memiliki skala jauh lebih besar dibandingkan fitur lainnya. Proses ini penting dilakukan terutama pada model _machine learning_ yang sensitif terhadap skala besar seperti KNN dan SVM. Dengan melakukan standarisasi, model dapat bekerja dengan lebih optimal dan menghasilkan performa yang lebih baik. 

Langkah pertama, membuat objek `StandarScaler()` yang akan digunakan untuk melakukan standarisasi data. Selanjutnya, fungsi `fit_transform()` diterapkan pada data training `x_train` untuk menghitung nilai rata-rata dan standar deviasi sekaligus melakukan transformasi data. 

Pada data validation `x_val` dan data testing `x_test_final` digunakan fungsi `transform()` agar proses _scalling_ mengikuti parameter yang diperoleh dari data training. Hal ini dilakukan untuk mencegah terjadinya _data leakage_ dan menjaga konsistensi skala data. Hasil _scalling_ kemudian disimpan ke dalam variabel `x_train_scaled`, `x_val_scaled`, dan `test_scaled` yang siap digunakan pada proses pelatihan dan evaluasi model _machine learning_.
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
test_scaled = scaler.transform(x_test_final)
```

# Model KNearest Neighbor (KNN)
Pada tahap ini dilakukan pembuatan model klasifikasi menggunakan model KNearest Neighbor (KNN). KNN merupakan algoritma _machine learning_ yang bekerja dengan mengklasifikasikan data berdasarkan tetangga terdekat dari suatu data baru. Model dibuat menggunakan parameter `n_neighbors=5` yang berarti proses prediksi dilakukan berdasarkan 5 data tetangga terdekat. Kelas yang paling banyak muncul dari tetangga tersebut akan menajdi hasil prediksi model. 

Selanjutnya dilakukan proses tarining menggunakan data training yang telah melalui proses _feature scalling_ (`x_train_scaled` dan `y_train`). Setelah model berhasil dilatih, model digunakan untuk memprediksi data validation `x_val_scaled` dan hasil prediksi disimpa pada variabel `y_pred_knn`. 

Untuk mengetahui performa model, dilakukan evaluasi menggunakan `accuracy_score` dan `classification_report`. Nilai akurasi digunakan untuk melihat tingkat ketepatan prediksi model, sedangkan _classification report menampilkan metrik evaluasi seperti precison, recall, dan f1-score pada setiap kelas `quality`.
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train_scaled, y_train)

y_pred_knn = knn.predict(x_val_scaled)

print("Akurasi KNN:", accuracy_score(y_val, y_pred_knn))
print(classification_report(y_val, y_pred_knn))
```
Berdasarkan hasil evaluasi, model KNN memperoleh nilai akurasi sebesar 0,6395 atau 63.95% pada data validation. Hal ini menunjukkan bahwa model mampu memprediksi sebagian besar data dengan cukup baik, namun performa belum optimal.

Hasil `classification_report` menunjukkan bahwa model memiliki performa yang lebih baik pada kelas 5, 6, dan 7. Pada kelas 5, model memperoleh nilai precision 0.66, recall 0.77, dan f1-score 0.71 yang menunjukkan bahwa model cukup baik dalam mengenali data dengan kualitas wine kelas 5. Pada kelas 6, model memperoleh niali f1-score sebesar 0.61. Sedangkan pada kelas 7, model memperoleh niali f1-score sebesar 0.64. Namun model tidak mampu memprediksi dengan baik pada kelas 3, 4, dan 8 yang ditinjukkan dengan nilai precision, recall, dan f1-score sebesar 0.00. Hal ini terjadi karena jumlah data pada kelas tersebut sangat sedikit dibanding kelas lainnya sehingga model kesulitan mengenali pola dari kelas minoritas. 

Sehingga secara keseluruhan, model KNN sudah mampu melakukan klasifikasi dengan cukup baik pada kelas mayoritas, namun perofrmanya masih kurang optimal dalam menangani kelas dengan jumlah data yang sedikit _(imbalanced class)_.

Pada tahap ini dilakukan evaluasi model KNN menggunakan _confusion matrix_ untuk melihat hasil prediksi model secara lebih detail pada setiap kelas `quality`. _Confusion matrix_ digunakan untuk membandingkan antara nilai aktual dengan nilai hasil prediksi model.
```python
cm_knn = confusion_matrix(y_val, y_pred_knn)

print("Confusion Matrix KNN:")
print(cm_knn)
```
Berdasarkan hasil output, diperoleh baris pada _confusion matrix_ menunjukkan kelas aktual, sedangkan kolom menunjukkan kelas hasil prediksi model. Nilai diagonal utama menunjukkan jumlah data yang berhasil diprediksi dengan benar. Pada kelas 5, terdapat 56 data yang berhasil diprediksi dengan benar, namun masih terdapat beberapa data yang salah diprediksi ke kelas 4 dan 6. Pada kelas 6, model berhasil memprediksi 40 data benar, tetapi terdapat beberapa data yang diprediksi sebagai kelas 5 dan 7. Pada kelas 7, terrdapat 14 data yang berhasil diprediksi dengan benar. Sedangkan kelas 3, 4, dan 8 model tidak mampu melakukan prediksi dengan baik karena jumlah data pada kelas tersebut sangat sedikit. Hal ini menyebabkan model cenderung lebih fokus pada kelas mayoritas seperti kelas 5 dan 6.

Secara keseluruhan, _consufion matrix_ menunjukkan bahwa model KNN cukup baik dalma memprediksi kelas mayoritas, namun masih mengalami kesalahan kalsifikasi pada beberapa kelas dan kurang optimal dalam menangani data dengan distribusi kelas yang tidak seimbang.

Pada tahap ini dilakukan proses visualisai _confusion matrix_ menggunakan `heatmap` dari librari `seabron`. Visualisasi ini bertujuan untuk mempermudah dalam melihat hasil klasifikasi model KNN pada setiap kelas `quality`. Fungsi `sns.heatmap()` digunakan utnuk menampilkan _confusion matrix_ dalam bentuk warna sehingga perbedaan jumlah prediksi benar dan salah dapat terlihat dengan jelas. Parameter `annot=True` digunakan agar nilai pada setiap sel ditampilkan, sedangkan `fmt='d'` digunakan utnuk menampilkan data dalam format bilangan bulat. Sedangkan parameter `cmap'Purples'` digunakan untuk memberikan gradasi warna ungu pada heatmap sehingga visualisai menjadi lebih mudah dibaca. Label sumbu `Predicted Label` menunjukkan hasil prediksi model, sedangkan `Actual Label` menunjukkan kelas sebenarnya.
```python
plt.figure(figsize=(8,6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Purples')

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - KNN')

plt.show()
```
<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/7471c433-a8d5-4932-be8b-ea9410a22e2c" />
