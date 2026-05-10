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

# Logistic Regression
Pada tahap ini dilakukan pembuatan model klasifikasi menggunakan `Logistic Regression`. Model ini digunakan untuk memprediksi kelas `quality` berdasarkan hubungan antara fitur-fitur kimia wine dengan target yang tersedia pada dataset training. Model ini dibuat menggunakan parameter `max_iter=1000` agar proses pelatihan memiliki jumlah iterasi yang cukup sehingga model dapat mencapai kondisi konvergen secara optimal. Setelah model dilatih menggunakan data tarining yang telah melalui proses _feature scalling_, model kemudia digunakan untuk memprediksi data validation. Hasil prediksi tersebut dievaluasi menggunakan `accuracy_score` dan `classification_report` untuk mengetahui performa model.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

lr = LogisticRegression(max_iter=1000)

lr.fit(x_train_scaled, y_train)

y_pred_lr = lr.predict(x_val_scaled)

print("Akurasi Logistic Regression:", accuracy_score(y_val, y_pred_lr))
print(classification_report(y_val, y_pred_lr))
```
Berdasarkan hasil evaluasi, model Logistic Regression memperoleh nilai akurasi sebesar 0.6104 atau 61.04% pada data validatiom. Nilai ini menunjukkan bahwa model mampu melakukan klasifikasi dengan cukup baik, meskipun performanya masih belum optimal. Model menunjukkan performa terbaik pada kelas 5 dengan nilai prediction sebesar 0.69, recall sebesar 0.73, dan f1-score sebesar 0.71. Sedangkan pada kelas 6, diperoleh nilai f1-score sebesar 0.58 dan pada kelas 7 sebesar 0.53. Hal ini menunjukkan bahwa model cukup mampu mengenali kelas mayoritas pada dataset. Namun, model tidak berhasil memprediksi kelas 3, 4, dan 8, yang terlihat dari nilai precision, recall, dan f1-score sebesar 0.00. Kondisi ini terjadi karena jumlah data pada kelas tersebut sangat sedikit dibandingkan kelas lainnya, sehingga model kesulitan mempelajari pola dari kelas minoritas.

kelas mayoritas seperti 5 dan 6, namun masih kurang optimal dalam menangani distribusi data yang tidak seimbang _(imbalanced dataset)_.

Tahap selanjutnya adalah evaluasi model Logistic Regression menggunakan _confusion matrix_ untuk melihat hasil klasifikasi model secara lebih detail pada setiap kelas  `quality`. Confusion matrix digunakan untuk membandingkan antara kelas aktual dengan kelas hasil prediksi model.
```python
cm_lr = confusion_matrix(y_val, y_pred_lr)

print("Confusion Matrix Logistic Regression:")
print(cm_lr)
```
Berdasarkan hasil _confusion matrix_, diperoleh pada baris menunjukkan kelas aktual, sedangkan kolom menunjukkan hasil prediksi model. Nilai pada diagonal utama menunjukkan jumlah data yang berhasil diprediksi dengan benar oleh model. Pada kelas 5, model berhasil memprediksi 53 data dengan benar, namun masih terdapat beberapa data yang salah diprediksi sebagai kelas 4 dan 6. Pada kelas 6, terdapat 40 data yang berhasil diprediksi dengan benar, tetapi beberapa data masih salah diprediksi sebagai kelas 5 dan 7. Pada kelas 7, model berhasil memprediksi 12 data dengan benar, namun masih terdapat data yang salah diklasifikasikan ke kelas 6. Sedangkan pada kelas 3, 4, dan 8, model belum mampu melakukan klasifikasi dengan baik karena jumlah data pada kelas tersebut sangat sedikit. Akibatnya, model lebih cenderung memprediksi ke kelas mayoritas seperti 5 dan 6.

Secara keseluruhan, confusion matrix menunjukkan bahwa model Logistic Regression cukup baik dalam mengenali kelas mayoritas, namun masih mengalami kesalahan klasifikasi pada beberapa kelas dan kurang optimal dalam menangani data dengan distribusi kelas yang tidak seimbang.

Pada tahap ini dilakukan visualisasi confusion matrix menggunakan `heatmap` dari library `seaborn`. Visualisasi ini bertujuan untuk mempermudah dalam melihat performa model Logistic Regression pada setiap kelas `quality`. Fungsi `sns.heatmap()` digunakan untuk menampilkan confusion matrix dalam bentuk warna sehingga hasil prediksi benar dan kesalahan klasifikasi dapat terlihat dengan lebih jelas. Parameter `annot=True` digunakan untuk menampilkan nilai pada setiap sel, sedangkan `fmt='d'` digunakan agar data ditampilkan dalam format bilangan bulat.Selain itu, parameter `cmap='Reds'` digunakan untuk memberikan gradasi warna merah pada heatmap sehingga visualisasi menjadi lebih mudah dibaca. Label `Predicted Label` menunjukkan hasil prediksi model, sedangkan `Actual Label` menunjukkan kelas sebenarnya dari data.
```python
plt.figure(figsize=(8,6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds')

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - Logistic Regression')

plt.show()
```
<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/1be45c11-22be-4910-a13d-5e8b28832202" />

# Support Vector Machine (SVM)
Pada tahap ini digunakan algoritma Support Vector Machine (SVM) untuk membangun model klasifikasi kualitas wine. SVM merupakan algoritma machine learning yang bekerja dengan mencari hyperplane terbaik untuk memisahkan setiap kelas data sehingga proses klasifikasi dapat dilakukan dengan lebih optimal. Model dibuat menggunakan parameter `kernel='rbf'` _(Radial Basis Function)_. Kernel ini digunakan untuk menangani hubungan data yang bersifat non-linear sehingga model dapat mempelajari pola data yang lebih kompleks. Selanjutnya, model dilatih menggunakan data training yang telah melalui proses feature scaling, yaitu `x_train_scaled` dan `y_train`. Setelah proses training selesai, model digunakan untuk memprediksi data validation `x_val_scaled` dan hasil prediksi disimpan dalam variabel `y_pred_svm`. Untuk mengetahui performa model, dilakukan evaluasi menggunakan `accuracy_score` dan `classification_report`.
``` python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

svm = SVC(kernel='rbf')

svm.fit(x_train_scaled, y_train)

y_pred_svm = svm.predict(x_val_scaled)

print("Akurasi SVM:", accuracy_score(y_val, y_pred_svm))
print(classification_report(y_val, y_pred_svm))
```
Berdasarkan hasil evaluasi, model SVM memperoleh nilai akurasi sebesar 0.6337 atau sekitar 63.37% pada data validation. Nilai ini menunjukkan bahwa model mampu melakukan klasifikasi dengan cukup baik. Model menunjukkan performa terbaik pada kelas 5 dengan nilai precision sebesar 0.70, recall sebesaar 0.78, dan f1-score sebesar 0.74. Sedangkan pada kelas 6, model memperoleh nilai f1-score sebesar 0.61, dan pada kelas 7 sebesar 0.46. Hal ini menunjukkan bahwa model cukup baik dalam mengenali kelas mayoritas pada dataset. Namun, model belum mampu memprediksi kelas 3, 4, dan 8, yang terlihat dari nilai precision, recall, dan f1-score sebesar 0.00. Kondisi ini terjadi karena jumlah data pada kelas tersebut sangat sedikit dibandingkan kelas lainnya sehingga model kesulitan mempelajari pola pada kelas minoritas.

Secara keseluruhan, model SVM memiliki performa yang cukup baik dan sedikit lebih unggul dibanding Logistic Regression, terutama dalam memprediksi kelas mayoritas seperti 5 dan 6. Namun, model masih kurang optimal dalam menangani distribusi data yang tidak seimbang _(imbalanced dataset)_.

Pada tahap ini dilakukan evaluasi model Support Vector Machine (SVM) menggunakan confusion matrix untuk melihat hasil klasifikasi model secara lebih detail pada setiap kelas `quality`. _Confusion matrix_ digunakan untuk membandingkan antara kelas aktual dengan kelas hasil prediksi model.
```python
cm_svm = confusion_matrix(y_val, y_pred_svm)

print("Confusion Matrix SVM:")
print(cm_svm)
```
Baris pada confusion matrix menunjukkan kelas aktual, sedangkan kolom menunjukkan hasil prediksi model. Nilai pada diagonal utama menunjukkan jumlah data yang berhasil diprediksi dengan benar oleh model. Pada kelas 5, model berhasil memprediksi 57 data dengan benar, namun masih terdapat beberapa data yang salah diprediksi sebagai kelas 6. Pada kelas 6, terdapat 43 data yang berhasil diprediksi dengan benar, tetapi beberapa data masih salah diprediksi sebagai kelas 5 dan 7. Pada kelas 7, model berhasil memprediksi 9 data dengan benar, namun masih terdapat beberapa data yang salah diprediksi ke kelas 6.

Secara keseluruhan, _confusion matrix_ menunjukkan bahwa model SVM cukup baik dalam mengenali kelas mayoritas dan memiliki performa yang cukup stabil. Namun, model masih mengalami kesalahan klasifikasi pada beberapa kelas dan kurang optimal dalam menangani data dengan distribusi kelas yang tidak seimbang.

Pada tahap ini dilakukan visualisasi confusion matrix dari model Support Vector Machine (SVM) menggunakan `heatmap` dari library `seaborn`. Visualisasi ini bertujuan untuk mempermudah dalam melihat hasil klasifikasi model pada setiap kelas `quality`. Fungsi `sns.heatmap()` digunakan untuk menampilkan _confusion matrix_ dalam bentuk warna sehingga hasil prediksi benar dan kesalahan klasifikasi dapat terlihat lebih jelas. Parameter `annot=True` digunakan agar nilai pada setiap sel ditampilkan, sedangkan `fmt='d'` digunakan untuk menampilkan data dalam format bilangan bulat. Selain itu, parameter `cmap='Blues'` digunakan untuk memberikan gradasi warna biru pada heatmap sehingga visualisasi menjadi lebih mudah dibaca. Label `Predicted Label` menunjukkan hasil prediksi model, sedangkan `Actual Label` menunjukkan kelas sebenarnya dari data.
<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/c0d0a94c-2976-4656-8395-bb7bba7e4039" />

# Random Forest
Pada tahap ini digunakan algoritma `RandomForestClassifier` untuk membangun model klasifikasi kualitas wine. Random Forest merupakan algoritma machine learning berbasis _decision tree_ yang bekerja dengan menggabungkan banyak pohon keputusan _(tree)_ untuk menghasilkan prediksi yang lebih akurat dan stabil. Model dibuat menggunakan parameter `n_estimators=100`, yang berarti model akan membangun 100 _decision tree_ dalam proses pelatihan. Selain itu, parameter `random_state=42` digunakan agar hasil model tetap konsisten setiap kali program dijalankan. Selanjutnya, model dilatih menggunakan data training yang telah melalui proses feature scaling, yaitu `x_train_scaled` dan `y_train`. Setelah proses training selesai, model digunakan untuk memprediksi data validation `x_val_scaled` dan hasil prediksi disimpan dalam variabel `y_pred_rf`. Untuk mengetahui performa model, dilakukan evaluasi menggunakan `accuracy_score` dan `classification_report`.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(x_train_scaled, y_train)

y_pred_rf = rf.predict(x_val_scaled)

print("Akurasi Random Forest:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))
```
Berdasarkan hasil evaluasi, model Random Forest memperoleh nilai akurasi sebesar 0.8721 atau sekitar 87.21% pada data validation. Nilai ini menunjukkan bahwa model memiliki performa yang sangat baik dalam melakukan klasifikasi kualitas wine dibandingkan model sebelumnya. Model menunjukkan performa yang tinggi pada hampir seluruh kelas `quality`. Pada kelas 5, model memperoleh nilai precision 0.87, recall 0.90, dan f1-score 0.89. Pada kelas 6, model memperoleh f1-score sebesar 0.85. Pada kelas 7, model memperoleh f1-score sebesar 0.89. Selain itu, model juga mampu memprediksi kelas minoritas seperti 3, 4, dan 8 dengan cukup baik, yang terlihat dari nilai precision dan recall yang tinggi pada kelas tersebut. Nilai _macro average_ sebesar 0.89 dan _weighted average_ sebesar 0.87 menunjukkan bahwa model memiliki performa yang stabil dan seimbang pada seluruh kelas. 

Secara keseluruhan, Random Forest menjadi model dengan performa terbaik dibandingkan algoritma sebelumnya karena mampu menghasilkan akurasi tinggi dan menangani distribusi kelas dengan lebih baik.

Pada tahap ini dilakukan evaluasi model Random Forest menggunakan _confusion matrix_ untuk melihat hasil klasifikasi model secara lebih detail pada setiap kelas `quality`. _Confusion matrix` digunakan untuk membandingkan antara kelas aktual dengan hasil prediksi model.
```python
cm_rf = confusion_matrix(y_val, y_pred_rf)

print("Confusion Matrix Random Forest:")
print(cm_rf)
```
Baris pada confusion matrix menunjukkan kelas aktual, sedangkan kolom menunjukkan hasil prediksi model. Nilai pada diagonal utama menunjukkan jumlah data yang berhasil diprediksi dengan benar oleh model. Berdasarkan hasil confusion matrix, terlihat bahwa model Random Forest mampu melakukan klasifikasi dengan cukup baik pada sebagian besar kelas `quality`. Pada kelas 5, sebagian besar data berhasil diprediksi dengan benar, meskipun masih terdapat beberapa data yang salah diprediksi ke kelas lain. Pada kelas 6, model berhasil memprediksi 57 data dengan benar, namun masih terdapat beberapa data yang salah diprediksi sebagai kelas 5 dan 7. Pada kelas 7, terdapat 20 data yang berhasil diprediksi dengan benar, menunjukkan performa model yang cukup baik pada kelas tersebut. Selain itu, jumlah kesalahan klasifikasi pada model Random Forest relatif lebih sedikit dibandingkan model sebelumnya seperti KNN, Logistic Regression, dan SVM. Hal ini menunjukkan bahwa Random Forest memiliki kemampuan yang lebih baik dalam mempelajari pola data dan menangani distribusi kelas yang berbeda-beda.

Secara keseluruhan, confusion matrix menunjukkan bahwa model Random Forest memiliki performa klasifikasi yang baik dan mampu menghasilkan prediksi yang lebih akurat dibandingkan algoritma lainnya.

Pada tahap ini dilakukan visualisasi confusion matrix dari model Random Forest menggunakan `heatmap` dari library `seaborn`. Visualisasi ini bertujuan untuk mempermudah dalam melihat performa model pada setiap kelas `quality`. Variabel labels digunakan untuk mengambil dan mengurutkan seluruh kelas unik pada target `quality`. Selanjutnya, _confusion matrix_ divisualisasikan menggunakan fungsi `sns.heatmap()` sehingga hasil klasifikasi dapat ditampilkan dalam bentuk warna dan angka yang lebih mudah dipahami. Parameter `annot=True` digunakan agar nilai pada setiap sel ditampilkan, sedangkan `fmt='d'` digunakan untuk menampilkan data dalam format bilangan bulat. Selain itu, parameter `cmap='Oranges'` digunakan untuk memberikan gradasi warna oranye pada heatmap sehingga visualisasi menjadi lebih jelas dan menarik. Label `Predicted Label` menunjukkan hasil prediksi model, sedangkan `Actual Label` menunjukkan kelas sebenarnya dari data.

<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/5360aaec-aafe-4066-b7f6-5946524b9c7a" />

# Decision Tree
Pada tahap ini digunakan algoritma  `DecisionTreeClassifier` untuk membangun model klasifikasi kualitas wine. Decision Tree merupakan algoritma machine learning yang bekerja dengan membentuk struktur pohon keputusan berdasarkan fitur-fitur pada dataset untuk menentukan hasil klasifikasi. Model dibuat menggunakan parameter  `random_state=42` agar hasil pelatihan model tetap konsisten setiap kali program dijalankan. Selanjutnya, model dilatih menggunakan data training `x_train` dan `y_train`. Setelah proses training selesai, model digunakan untuk memprediksi data validation `x_val` dan hasil prediksi disimpan dalam variabel `y_pred_dt`. Untuk mengetahui performa model, dilakukan evaluasi menggunakan `accuracy_score` dan `classification_report`.
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dt = DecisionTreeClassifier(random_state=42)

dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_val)

print("Akurasi Decision Tree:", accuracy_score(y_val, y_pred_dt))
print(classification_report(y_val, y_pred_dt))
```
Berdasarkan hasil evaluasi, model Decision Tree memperoleh nilai akurasi sebesar 0.8547 atau sekitar 85.47% pada data validation. Nilai ini menunjukkan bahwa model memiliki performa yang sangat baik dalam melakukan klasifikasi kualitas wine. Model menunjukkan performa yang cukup tinggi pada hampir seluruh kelas `quality`. Pada kelas 5, model memperoleh nilai precision 0.90, recall 0.85, dan f1-score 0.87. Pada kelas 6, model memperoleh f1-score sebesar 0.85. Pada kelas 7, model memperoleh f1-score sebesar 0.88, yang menunjukkan bahwa model sangat baik dalam mengenali kelas tersebut. Selain itu, model juga mampu memprediksi kelas minoritas seperti 3, 4, dan 8 dengan cukup baik dibandingkan beberapa model sebelumnya. Nilai weighted average f1-score sebesar 0.86 menunjukkan bahwa model memiliki performa klasifikasi yang stabil pada seluruh kelas.

Secara keseluruhan, Decision Tree menjadi salah satu model dengan performa terbaik pada proyek ini karena mampu menghasilkan akurasi tinggi dan menangani data klasifikasi dengan cukup baik.

Pada tahap ini dilakukan evaluasi model Decision Tree menggunakan _confusion matrix_ untuk melihat hasil klasifikasi model secara lebih detail pada setiap kelas `quality`. _Confusion matrix_ digunakan untuk membandingkan antara kelas aktual dengan hasil prediksi model.
```python
cm_dt = confusion_matrix(y_val, y_pred_dt)

print("Confusion Matrix Decision Tree:")
print(cm_dt)
```
Baris pada confusion matrix menunjukkan kelas aktual, sedangkan kolom menunjukkan hasil prediksi model. Nilai pada diagonal utama menunjukkan jumlah data yang berhasil diprediksi dengan benar oleh model. Pada kelas 3, model berhasil memprediksi 1 data dengan benar tanpa kesalahan klasifikasi. Pada kelas 4, terdapat 4 data yang berhasil diprediksi dengan benar, namun masih terdapat 1 data yang salah diprediksi sebagai kelas 8. Pada kelas 5, model berhasil memprediksi 62 data dengan benar, meskipun masih terdapat beberapa data yang salah diprediksi ke kelas 4, 6, dan 7. Pada kelas 6, terdapat 57 data yang berhasil diprediksi dengan benar, sedangkan beberapa data masih salah diprediksi sebagai kelas lain. Pada kelas 7, model berhasil memprediksi 21 data dengan benar dan hanya sedikit kesalahan klasifikasi. Pada kelas 8, terdapat 2 data yang berhasil diprediksi dengan benar dan 1 data salah diprediksi sebagai kelas 7.

Secara keseluruhan, _confusion matrix_ menunjukkan bahwa model Decision Tree memiliki performa yang baik dalam mengenali sebagian besar kelas  `quality`. Sebagian besar nilai berada pada diagonal utama, yang menandakan bahwa model mampu melakukan klasifikasi dengan cukup akurat. Namun, masih terdapat beberapa kesalahan prediksi terutama pada kelas yang memiliki karakteristik yang mirip satu sama lain.

Pada tahap ini dilakukan visualisasi confusion matrix dari model Decision Tree menggunakan `heatmap` dari library `seaborn`. Visualisasi ini bertujuan untuk mempermudah dalam melihat performa model pada setiap kelas `quality`. Fungsi `sns.heatmap()` digunakan untuk menampilkan _confusion matrix_ dalam bentuk warna sehingga hasil klasifikasi benar dan kesalahan prediksi dapat terlihat lebih jelas. Parameter `annot=True` digunakan agar nilai pada setiap sel ditampilkan, sedangkan `fmt='d'` digunakan untuk menampilkan data dalam format bilangan bulat. Selain itu, parameter `cmap='Greens'` digunakan untuk memberikan gradasi warna hijau pada heatmap sehingga visualisasi menjadi lebih mudah dibaca. Label `Predicted Label` menunjukkan hasil prediksi model, sedangkan `Actual Label` menunjukkan kelas sebenarnya dari data.
<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/0ec0d340-7b86-464b-8f1f-5849e2b69345" />
```python
plt.figure(figsize=(8,6))

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens')

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - Decision Tree')

plt.show()
```

Pada tahap ini dilakukan visualisasi struktur pohon keputusan _(decision tree)_ dari model `DecisionTreeClassifier` menggunakan fungsi `plot_tree()` dari `scikit-learn`. Visualisasi ini bertujuan untuk melihat bagaimana model mengambil keputusan dalam proses klasifikasi kualitas wine berdasarkan fitur-fitur yang tersedia. Fungsi `dt.fit(x_train, y_train)` digunakan untuk melatih model Decision Tree menggunakan data training. Selanjutnya, fungsi `plot_tree()` digunakan untuk menampilkan struktur pohon keputusan secara visual. Parameter `filled=True` digunakan agar setiap node pada pohon memiliki warna yang berbeda sesuai dengan kelas dominan pada node tersebut, sehingga visualisasi menjadi lebih mudah dipahami. Selain itu, ukuran gambar diperbesar menggunakan `figsize=(75,50)` agar struktur pohon dapat terlihat dengan lebih jelas karena jumlah node yang cukup banyak.
<img width="5832" height="3870" alt="image" src="https://github.com/user-attachments/assets/47434ca2-260d-4eba-b7fd-134260a0f451" />

# Membanding Akurasi Model
Pada tahap ini dilakukan perbandingan performa beberapa algoritma _machine learning_ yang telah digunakan pada proses klasifikasi kualitas wine. Perbandingan dilakukan dengan menghitung nilai akurasi dari masing-masing model menggunakan fungsi `accuracy_score()`.

Nilai akurasi dari setiap model kemudian disimpan ke dalam variabel:
`akurasi_lr` untuk Logistic Regression
`akurasi_knn` untuk KNN
`akurasi_svm` untuk SVM
`akurasi_rf` untuk Random Forest
`akurasi_dt` untuk Decision Tree

Selanjutnya, seluruh hasil akurasi digabungkan ke dalam sebuah `DataFrame` bernama `hasil_model` agar lebih mudah dibandingkan dan dianalisis.
```python
akurasi_lr = accuracy_score(y_val, y_pred_lr)
akurasi_knn = accuracy_score(y_val, y_pred_knn)
akurasi_svm = accuracy_score(y_val, y_pred_svm)
akurasi_rf = accuracy_score(y_val, y_pred_rf)
akurasi_dt = accuracy_score(y_val, y_pred_dt)

hasil_model = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVM', 'Random Forest', 'Decision Tree'],
    'Akurasi': [akurasi_lr, akurasi_knn, akurasi_svm, akurasi_rf, akurasi_dt]
})

hasil_model
```
## 📊 Perbandingan Akurasi Model

| No | Model                  | Akurasi |
|----|------------------------|----------|
| 1  | Logistic Regression    | 0.6105 |
| 2  | KNN                    | 0.6395 |
| 3  | SVM                    | 0.6337 |
| 4  | Random Forest          | 0.8721 |
| 5  | Decision Tree          | 0.8547 |

Berdasarkan tabel di atas, model Random Forest memperoleh nilai akurasi tertinggi yaitu sekitar 87.21%, sehingga menjadi model dengan performa terbaik pada proyek ini. Model ini mampu melakukan klasifikasi dengan lebih baik dibandingkan algoritma lainnya karena dapat menangani pola data yang kompleks dan mengurangi risiko overfitting melalui penggunaan banyak decision tree. Model Decision Tree juga menunjukkan performa yang sangat baik dengan akurasi sekitar 85.47%. Sementara itu, model KNN, SVM, dan Logistic Regression memiliki akurasi yang lebih rendah, yaitu berada pada rentang 61%–64%.

Secara keseluruhan, hasil evaluasi menunjukkan bahwa algoritma berbasis pohon keputusan seperti Random Forest dan Decision Tree lebih efektif digunakan pada dataset klasifikasi kualitas wine dibandingkan algoritma lainnya pada proyek ini.

# Menggunakan Random Forest
Pada tahap ini dipilih algoritma `RandomForestClassifier` sebagai model terbaik berdasarkan hasil evaluasi sebelumnya yang menunjukkan bahwa Random Forest memiliki nilai akurasi paling tinggi dibandingkan model lainnya. Model dibuat menggunakan parameter pertama`n_estimators=100` yang berarti model akan membangun 100 decision tree untuk meningkatkan performa klasifikasi. `random_state=42` digunakan agar hasil pelatihan model tetap konsisten setiap kali program dijalankan. `class_weight='balanced'` digunakan untuk menangani distribusi kelas yang tidak seimbang _(imbalanced dataset)_ sehingga model dapat memberikan perhatian yang lebih seimbang pada setiap kelas `quality`.

Selanjutnya, model dilatih menggunakan seluruh data training yang telah melalui proses feature scaling dengan `scaler.transform(x)`. Penggunaan seluruh data training bertujuan agar model dapat mempelajari pola data secara maksimal sebelum digunakan untuk melakukan prediksi pada dataset testing.
```python
model_terbaik = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

model_terbaik.fit(scaler.transform(x), y)
```
Dengan proses ini, model Random Forest siap digunakan sebagai model final untuk memprediksi kualitas wine pada dataset testing.

# 4. Prediksi Hasil
Pada tahap ini dilakukan proses prediksi kualitas wine pada dataset testing menggunakan model terbaik, yaitu Random Forest. Model yang telah dilatih sebelumnya digunakan untuk memprediksi nilai `quality` berdasarkan fitur-fitur kimia pada data testing. Fungsi `predict()` digunakan untuk menghasilkan prediksi kualitas wine dari data testing yang telah melalui proses feature scaling `test_scaled`. Hasil prediksi kemudian disimpan dalam variabel `prediksi_quality`. Selanjutnya, dibuat sebuah DataFrame bernama `hasil_prediksi` yang berisi Kolom `Id` sebagai identitas data wine pada dataset testing dan kolom `quality` sebagai hasil prediksi kualitas wine dari model machine learning.
```python
prediksi_quality = model_terbaik.predict(test_scaled)

hasil_prediksi = pd.DataFrame({
    'Id': data_testing['Id'],
    'quality': prediksi_quality
})

hasil_prediksi
```

Hasil Prediksi Quality Wine

| No | Id   | quality |
|----|------|----------|
| 0  | 222  | 5 |
| 1  | 1514 | 6 |
| 2  | 417  | 5 |
| 3  | 754  | 5 |
| 4  | 516  | 5 |
| ... | ... | ... |
| 281 | 1147 | 6 |
| 282 | 296  | 5 |
| 283 | 170  | 5 |
| 284 | 1439 | 6 |
| 285 | 946  | 7 |

Tabel di atas menunjukkan hasil prediksi kualitas wine pada dataset testing menggunakan model Random Forest. Kolom `quality` berisi hasil klasifikasi kualitas wine yang diprediksi berdasarkan pola yang telah dipelajari dari dataset training.

# Menyimpan Hasil
Pada tahap ini dilakukan penyimpanan hasil prediksi kualitas wine ke dalam file berformat CSV menggunakan fungsi `to_csv()` dari library `pandas`. File hasil prediksi disimpan dengan nama `hasilprediksi_007.csv`. File ini berisi kolom `Id` dan `quality`, yaitu identitas data wine serta hasil prediksi kualitas wine yang dihasilkan oleh model machine learning. Parameter `index=False` digunakan agar index bawaan dari `DataFrame` tidak ikut tersimpan ke dalam file CSV sehingga hasil file menjadi lebih rapi dan sesuai format yang dibutuhkan.
```python
hasil_prediksi.to_csv('hasilprediksi_007.csv', index=False)
```

# Insight dari Analisis
Berdasarkan hasil analisis dan pengujian beberapa algoritma machine learning, diketahui bahwa kualitas wine dapat diprediksi menggunakan fitur-fitur kimia seperti `alcohol`, `pH`, `sulphates`, `volatile acidity`, dan fitur lainnya. Dataset yang digunakan memiliki distribusi kelas yang tidak seimbang, di mana kelas 5 dan 6 memiliki jumlah data paling banyak dibandingkan kelas lainnya. Hal ini memengaruhi performa beberapa model seperti Logistic Regression, KNN, dan SVM yang cenderung lebih baik dalam memprediksi kelas mayoritas dibandingkan kelas minoritas.

Dari seluruh model yang diuji, algoritma berbasis pohon keputusan seperti Random Forest dan Decision Tree menunjukkan performa terbaik dibandingkan model lainnya. Random Forest berhasil memperoleh akurasi tertinggi sebesar sekitar 87%, yang menunjukkan bahwa model mampu mempelajari pola data dengan baik dan menghasilkan prediksi yang lebih stabil. Selain itu, confusion matrix menunjukkan bahwa sebagian besar data berhasil diklasifikasikan dengan benar, terutama pada kelas 5, 6, dan 7.

# Kesimpulan
Berdasarkan hasil analisis yang telah dilakukan, dapat disimpulkan bahwa proses klasifikasi kualitas wine berhasil dilakukan menggunakan beberapa algoritma machine learning seperti Logistic Regression, KNN, SVM, Random Forest, dan Decision Tree.

Hasil evaluasi menunjukkan bahwa model Random Forest menjadi model terbaik dengan akurasi sekitar 87.21%, diikuti oleh Decision Tree dengan akurasi sekitar 85.47%. Sementara itu, model Logistic Regression, KNN, dan SVM memiliki performa yang lebih rendah karena kurang optimal dalam menangani distribusi data yang tidak seimbang.

Model terbaik kemudian digunakan untuk memprediksi kualitas wine pada dataset testing dan hasil prediksi berhasil disimpan ke dalam file CSV. Dengan demikian, project ini berhasil membangun model machine learning yang mampu memprediksi kualitas wine secara otomatis berdasarkan karakteristik kimia yang dimiliki wine.

# Rekomendasi
Berdasarkan hasil project yang telah dilakukan, terdapat beberapa rekomendasi yang dapat dikembangkan pada penelitian atau project selanjutnya, yaitu:
1. Melakukan penanganan imbalanced dataset menggunakan teknik seperti SMOTE atau oversampling agar model dapat memprediksi kelas minoritas dengan lebih baik.
2. Melakukan hyperparameter tuning pada setiap algoritma machine learning untuk memperoleh performa model yang lebih optimal.
3. Menambahkan proses feature selection atau feature engineering untuk mengetahui fitur yang paling berpengaruh terhadap kualitas wine.
4. Menggunakan dataset dengan jumlah data yang lebih besar agar model dapat mempelajari pola data dengan lebih baik dan menghasilkan prediksi yang lebih akurat.
5. Mengembangkan model ke dalam bentuk aplikasi atau deployment sehingga dapat digunakan secara langsung untuk memprediksi kualitas wine secara real-time.
