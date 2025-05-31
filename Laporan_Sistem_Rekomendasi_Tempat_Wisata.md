# Laporan Proyek Machine Learning – Rekomendasi Wisata Kota Bandung

![Ikon Kota Bandung](assets/iconic_bandung.jpeg)

## Project Overview

Pada era globalisasi, sektor pariwisata telah menjadi salah satu industri terbesar di dunia. Menurut **[Prawita, Yanuar, dan Sabani (2020)](https://openlibrarypublications.telkomuniversity.ac.id/index.php/appliedscience/article/viewFile/13937/13677)**, secara kumulatif pada Juli 2015, jumlah kunjungan wisatawan mancanegara ke Indonesia mencapai 814.233 kunjungan, meningkat dibandingkan periode yang sama tahun sebelumnya yang berjumlah 777.210 orang. Fakta ini menunjukkan bahwa sektor pariwisata merupakan salah satu sektor yang sangat menarik untuk terus dikembangkan oleh suatu negara. Pariwisata dipandang sebagai aset strategis yang mampu mendorong pembangunan di wilayah-wilayah dengan potensi objek wisata. Kota Bandung, sebagai salah satu kota utama di Jawa Barat, memiliki keanekaragaman budaya yang menjadi kekayaan daerah dan perlu dikembangkan secara optimal. Bandung juga menawarkan berbagai objek wisata budaya dan sejarah yang menarik untuk dikunjungi. Namun, wisatawan sering menghadapi kesulitan dalam menentukan destinasi yang sesuai dengan minat mereka, terutama bagi mereka yang telah beberapa kali berkunjung dan ingin menemukan pengalaman baru.

Berbagai penelitian telah mengembangkan pendekatan untuk mengatasi masalah ini. Menurut **[Pasaribu dan Sitompul (2023)](https://journal.staiypiqbaubau.ac.id/index.php/Mutiara/article/download/736/796)**, sistem rekomendasi destinasi wisata di Bandung dapat dibangun menggunakan algoritma *Collaborative Filtering* (CF) untuk membantu wisatawan memilih tempat wisata sesuai preferensi mereka. Sistem ini memanfaatkan data pengguna, rating destinasi, serta informasi wisata untuk membangun model prediksi berbasis *machine learning* menggunakan *RecommenderNet* dari TensorFlow. Hasilnya mampu menghasilkan rekomendasi yang lebih personal, meningkatkan pengalaman wisatawan, serta mendukung pengembangan sektor pariwisata di Bandung. Penelitian ini juga menyarankan pengembangan model *hybrid* untuk meningkatkan akurasi dengan mempertimbangkan faktor tambahan seperti biaya, waktu, dan jarak tempuh.

Sementara itu, **[Adlan dan Setiawan (2025)](https://teknokom.unwir.ac.id/index.php/teknokom/article/view/152)** mengembangkan sistem rekomendasi destinasi wisata di Bandung dengan pendekatan *user-based collaborative filtering* menggunakan algoritma *K-Nearest Neighbors* (KNN) serta dua metode pengukuran kesamaan, yaitu *cosine similarity* dan *pearson correlation*. Dataset yang digunakan diambil dari Kaggle dan difokuskan pada destinasi wisata di Bandung. Evaluasi performa menunjukkan bahwa metode *cosine similarity* memberikan nilai *Mean Absolute Error* (MAE) sebesar 2,59 pada K=3, sementara *pearson correlation* menghasilkan MAE sebesar 2,67 pada K=30. Penelitian ini menyimpulkan bahwa kombinasi collaborative filtering dengan klasifikasi KNN mampu mengatasi masalah *data sparsity* dan menghasilkan rekomendasi yang lebih relevan dan personal.

Berdasarkan landasan tersebut, proyek ini bertujuan untuk membangun sistem rekomendasi wisata di Kota Bandung yang tidak hanya cerdas dan akurat, tetapi juga memperhatikan konteks lokal, tren wisata berbasis pengalaman, serta keberlanjutan industri pariwisata kreatif. Sistem ini diharapkan dapat meningkatkan pengalaman pengguna, mempercepat proses pengambilan keputusan wisata, dan mendorong pemerataan kunjungan wisata ke berbagai titik potensi di Kota Bandung.

---

## Business Understanding

### Problem Statement

Sebagai salah satu destinasi wisata utama di Indonesia, Bandung menawarkan ratusan pilihan tempat wisata dari berbagai kategori, seperti alam, budaya, sejarah, hingga kuliner. Meski begitu, wisatawan baik yang baru pertama kali berkunjung maupun yang sudah beberapa kali datang sering menghadapi kesulitan dalam menentukan destinasi yang paling sesuai dengan preferensi dan minat mereka. Tidak adanya sistem rekomendasi yang dipersonalisasi menyebabkan proses pemilihan destinasi menjadi kurang efisien, sehingga pengalaman berwisata tidak maksimal. Selain itu, hal ini juga dapat menghambat potensi pertumbuhan ekonomi bagi pelaku wisata lokal karena distribusi kunjungan wisatawan yang tidak merata, terutama ke destinasi yang kurang terkenal namun sebenarnya memiliki daya tarik.

### Goals

Proyek ini memiliki beberapa tujuan, yaitu:

* Merancang sistem rekomendasi wisata yang mampu **menyajikan destinasi secara lebih relevan dan personal** sesuai preferensi masing-masing pengguna.
* Membangun serta melakukan perbandingan antara **dua pendekatan model rekomendasi**, yakni Content-Based Filtering (CBF) dan Collaborative Filtering (CF).
* Mengevaluasi performa kedua model menggunakan metrik seperti **top-N recommendation accuracy (Precision\@K)** dan **Root Mean Square Error (RMSE)** guna memastikan kualitas prediksi serta relevansi rekomendasi yang dihasilkan.

### Solution Statement

Solusi yang ditawarkan mencakup dua pendekatan:

* **Content-Based Filtering (CBF):** Mengandalkan kesamaan antara deskripsi serta kategori tempat wisata dengan memanfaatkan teknik TF-IDF dan cosine similarity.
* **Collaborative Filtering (CF):** Menggunakan model matrix factorization berbasis embedding yang dilatih menggunakan data interaksi antara pengguna dan destinasi, sambil mempertimbangkan elemen kepercayaan seperti jumlah tetangga, rata-rata rating pengguna, dan rating destinasi.

---

## Data Understanding

### Dataset

Penelitian ini memanfaatkan tiga dataset yang saling terhubung untuk membangun sistem rekomendasi tempat wisata:

* `tourism_with_id.csv`: berisi informasi detail mengenai masing-masing tempat wisata.
* `tourism_rating.csv`: memuat data interaksi berupa rating yang diberikan pengguna terhadap tempat wisata.
* `user.csv`: mencakup data demografi para pengguna sistem.

**Sumber dataset**: [Kaggle - Indonesia Tourism Destination Dataset](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

### Struktur dan Kondisi Data

#### 1. Dataset `tourism_with_id.csv`

**Jumlah data:** 437 baris × 13 kolom

**Penjelasan fitur:**
* `Place_Id`: ID unik untuk masing-masing tempat wisata (integer)
* `Place_Name`: Nama tempat wisata (object)
* `Description`: Deskripsi lengkap mengenai tempat wisata (object)
* `Category`: Kategori destinasi, misalnya Budaya, Alam, Bahari, dll. (object)
* `City`: Nama kota lokasi tempat wisata berada (object)
* `Price`: Harga tiket masuk dalam satuan rupiah (integer)
* `Rating`: Rating rata-rata tempat wisata berdasarkan ulasan pengunjung dalam skala 1-5 (float)
* `Time_Minutes`: Estimasi waktu yang dibutuhkan untuk mengunjungi tempat wisata (menit, float)
* `Coordinate`: Koordinat lokasi dalam format object, sebenarnya redundan karena sudah ada kolom `Lat` dan `Long` (object)
* `Lat`: Koordinat lintang lokasi destinasi (float)
* `Long`: Koordinat bujur lokasi destinasi (float)
* `Unnamed: 11`: Kolom kosong tanpa data yang relevan (float)
* `Unnamed: 12`: Kolom tambahan berupa angka, namun tidak memiliki kejelasan fungsi (integer)

**Representasi dalam bentuk tabel**

| Fitur          | Tipe Data               | Deskripsi                                                             |
| -------------- | ----------------------- | --------------------------------------------------------------------- |
| `Place_Id`     | Integer                 | ID unik untuk tempat wisata                                           |
| `Place_Name`   | object                  | Nama tempat wisata                                                    |
| `Description`  | object                  | Deskripsi tempat wisata                                               |
| `Category`     | object                  | Kategori wisata (Budaya, Alam, dll.)                                  |
| `City`         | object                  | Kota lokasi tempat wisata                                             |
| `Price`        | Integer                 | Harga tiket masuk (rupiah)                                            |
| `Rating`       | Float                   | Rating rata-rata pengunjung (skala 1–5)                               |
| `Time_Minutes` | Float                   | Estimasi waktu kunjungan (menit) – banyak missing value (53%)         |
| `Coordinate`   | object                  | Koordinat dalam format object (string), redundan dengan kolom `Lat` dan `Long` |
| `Lat`          | Float                   | Koordinat lintang                                                     |
| `Long`         | Float                   | Koordinat bujur                                                       |
| `Unnamed: 11`  | Float (kosong)          | Kolom tanpa data (seluruh nilai kosong)                               |
| `Unnamed: 12`  | Integer (tidak relevan) | Kolom tambahan tanpa makna khusus                                     |

**Kondisi data:**
* **Missing values (nilai kosong):** `Time_Minutes` memiliki sekitar 232 missing values (53% dari total data), sementara `Unnamed: 11` seluruhnya kosong.
* **Kolom yang dihapus:** `Time_Minutes`, `Unnamed: 11`, dan `Unnamed: 12` dihilangkan dari analisis karena jumlah missing value yang besar atau karena tidak relevan bagi pengembangan sistem rekomendasi.
* **Duplikat:** Tidak ditemukan data duplikat.

#### 2. Dataset `tourism_rating.csv`

**Jumlah data:** 10.000 baris × 3 kolom

**Penjelasan fitur:**
* `User_Id`: ID unik untuk setiap pengguna yang memberikan penilaian (integer)
* `Place_Id`: ID tempat wisata yang dinilai, merujuk pada `Place_Id` di dataset tempat wisata (integer)
* `Place_Ratings`:  Nilai rating yang diberikan pengguna terhadap tempat wisata, dengan skala 1–5 (integer)

**Representasi dalam bentuk tabel**

| Fitur           | Tipe Data | Deskripsi                                                         |
| --------------- | --------- | ----------------------------------------------------------------- |
| `User_Id`       | Integer   | ID unik pengguna pemberi rating                                   |
| `Place_Id`      | Integer   | ID tempat wisata yang diberi rating (mengacu ke `Place_Id`)       |
| `Place_Ratings` | Integer   | Nilai rating dari pengguna untuk tempat wisata (skala 1–5)        |

**Kondisi data:**
* **Missing values (nilai kosong):** Tidak ditemukan nilai kosong.
* **Duplikat:** Terdapat **79 baris data duplikat**, namun data tersebut tetap dipertahankan karena:
  * **Rating bersifat dinamis**, seorang pengguna bisa memberikan penilaian lebih dari sekali pada satu destinasi, misalnya setelah kunjungan berikutnya.
  * Duplikasi ini mencerminkan **perubahan persepsi atau pengalaman pengguna dari waktu ke waktu**, sehingga tetap dianggap relevan dalam membentuk preferensi pengguna secara lebih akurat dalam sistem rekomendasi.

#### 3. Dataset `user.csv`

**Jumlah data:** 300 baris × 3 kolom

**Penjelasan fitur:**
* `User_Id`: ID unik yang merepresentasikan masing-masing pengguna, sesuai dengan `User_Id` pada dataset rating (integer)
* `Location`: Nama kota atau daerah asal pengguna (object)
* `Age`: Usia pengguna dalam satuan tahun (integer)

**Representasi dalam bentuk tabel**

| Fitur      | Tipe Data | Keterangan                    |
| ---------- | --------- | ----------------------------- |
| `User_Id`  | Integer   | ID unik untuk setiap pengguna |
| `Location` | Object    | Kota atau daerah asal pengguna|
| `Age`      | Integer   | Usia pengguna (tahun)         |

**Kondisi data:**
* **Missing values (nilai kosong):** Tidak ditemukan nilai kosong pada dataset ini.
* **Duplikat:** Tidak terdapat baris data yang duplikat.

### Insight Awal

* **Kelengkapan metadata:** Sebagian besar tempat wisata memiliki detail deskriptif yang memadai, namun data estimasi waktu kunjungan (`Time_Minutes`) memiliki banyak kekosongan sehingga tidak dapat dimanfaatkan secara optimal.

* **Kualitas data interaksi:** Data rating pengguna tergolong kaya dengan 10.000 catatan interaksi, yang dapat dimanfaatkan untuk membangun sistem rekomendasi menggunakan pendekatan Collaborative Filtering.

* **Profil pengguna:** Terdapat data dari 300 pengguna unik yang dilengkapi informasi demografis, memungkinkan analisis terkait preferensi berdasarkan usia dan lokasi.

* **Sebaran geografis:** Data tempat wisata mencakup destinasi dari berbagai kota di Indonesia serta mencakup beragam kategori, memberikan variasi yang baik untuk mendukung sistem rekomendasi.

### Visualisasi

![rating_pengunjung](https://github.com/user-attachments/assets/4145cc8f-6043-44fc-8435-0fa0816f2cf9)

*Gambar 1. Histogram persebaran rating yang diberikan oleh pengunjung.*

---

## Data Preparation

Tahap persiapan data termasuk salah satu proses penting untuk memastikan sistem rekomendasi dapat dibangun secara optimal. Berikut merupakan rangkuman tahapan yang dilakukan selama proses data preparation sebelum data digunakan dalam pemodelan Content-Based Filtering (CBF) maupun Collaborative Filtering (CF).

### Ringkasan Tahapan Data Preparation

| No | Tahapan                     | Deskripsi                                             | Tujuan Utama                                         |
| -- | --------------------------- | ----------------------------------------------------- | ---------------------------------------------------- |
| 1  | Data Cleaning               | Menghapus kolom yang tidak relevan                    | Meningkatkan kualitas data dan efisiensi pemrosesan  |
| 2  | Filter Lokasi               | Memfilter data agar hanya mencakup wilayah Bandung    | Menjaga relevansi geografis untuk hasil rekomendasi  |
| 3  | Merge Dataset               | Menggabungkan data rating, tempat, dan pengguna       | Memastikan konsistensi referensi antar tabel         |
| 4  | TF-IDF Vectorization        | Mengubah kategori menjadi representasi numerik        | Menyiapkan fitur untuk Content-Based Filtering       |
| 5  | Encoding ID Manual          | Mengonversi ID menjadi indeks numerik                 | Menyesuaikan format agar kompatibel dengan model ML  |
| 6  | Normalisasi Rating Manual   | Menstandarkan skala rating ke rentang 0–1             | Menjaga stabilitas pelatihan pada model neural net   |
| 7  | Random Shuffle Data         | Mengacak urutan data                                  | Menghindari bias dan memastikan distribusi merata    |
| 8  | Split Data Manual           | Memisahkan data untuk training dan validasi           | Mendukung evaluasi model secara objektif             |

### Detail Implementasi

#### 1. Data Cleaning (Pembersihan Data)

**Kode Program:**
```python
df_place = df_place.drop(['Time_Minutes','Unnamed: 11','Unnamed: 12'], axis=1)
```

**Proses yang Dilakukan:**
Menghapus kolom-kolom yang tidak relevan atau memiliki tingkat missing value yang tinggi, seperti Time_Minutes, Unnamed: 11, dan Unnamed: 12, dari dataset tempat wisata.

**Alasan Dilakukan:**
* Kolom dengan banyak missing value dapat mengganggu kualitas model dan berpotensi memicu error saat pelatihan
* Kolom yang tidak informatif hanya akan menambah noise tanpa kontribusi signifikan dalam proses pembelajaran
* Mengurangi dimensi yang tidak penting membantu meningkatkan efisiensi komputasi dan mempercepat proses pelatihan
* Data yang sudah dibersihkan memungkinkan sistem rekomendasi bekerja lebih akurat dan optimal

#### 2. Filter Lokasi: Bandung

**Kode Program:**
```python
df_place = df_place[df_place['City'] == 'Bandung']
```

**Proses yang Dilakukan:**
Melakukan penyaringan data pada dataset tempat wisata sehingga hanya destinasi yang berlokasi di Kota Bandung yang dipertahankan, berdasarkan informasi pada kolom City.

**Alasan Dilakukan:**
* Memusatkan sistem rekomendasi pada satu wilayah geografis untuk menjaga fokus dan arah yang jelas
* Mengurangi kompleksitas data dengan mempersempit lingkup area, sehingga model bisa bekerja lebih spesifik
* Menjamin hasil rekomendasi tetap relevan untuk pengguna yang berada atau berencana berwisata di Bandung
* Mengurangi risiko munculnya bias geografis yang dapat memengaruhi kualitas rekomendasi

#### 3. Merge Dataset (Penggabungan Dataset)

**Kode Program:**
```python
df_rating = pd.merge(df_rating, df_place[['Place_Id']], how='right', on='Place_Id')

df_user = pd.merge(df_user, df_rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')
```

**Proses yang Dilakukan:**
Menggabungkan dataset rating dengan dataset tempat wisata berdasarkan kolom Place_Id, serta menggabungkan data pengguna berdasarkan kolom User_Id. Setelah itu, data duplikat dihapus dan data diurutkan berdasarkan User_Id.

**Alasan Dilakukan:**
* Memastikan konsistensi antara tabel-tabel, sehingga tidak ada data yang tidak valid
* Menghapus data orphan seperti rating tanpa tempat wisata terkait atau pengguna yang tidak valid
* Menyediakan data yang bersih, lengkap, dan terintegrasi untuk proses pemodelan berikutnya
* Memastikan setiap data rating memiliki informasi data yang jelas, baik dari user maupun tempat wisata

#### 4. TF-IDF Vectorization

**Kode Program:**
```python
tfidf_vectorizer_for_category = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer_for_category.fit_transform(df_place['Category'])
```

**Proses yang Dilakukan:**
Menggunakan TfidfVectorizer pada kolom *Category* untuk mengonversi data kategori tempat wisata dari teks menjadi bentuk numerik berupa matriks TF-IDF.

**Alasan Dilakukan:**
* Mengubah data teks kategori menjadi format numerik agar dapat digunakan dalam algoritma machine learning
* TF-IDF memberi bobot lebih besar pada kata-kata unik yang jarang muncul namun informatif dalam membedakan kategori
* Memungkinkan perhitungan similarity berbasis konten yang diperlukan dalam Content-Based Filtering
* Representasi dalam bentuk vektor memungkinkan operasi matematis untuk menghitung tingkat kesamaan antar tempat wisata

#### 5. Encoding ID Manual

**Kode Program:**
```python
def dict_encoder(col, data=df):
    unique_val = data[col].unique().tolist()
    val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
    return val_to_val_encoded, val_encoded_to_val

user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id')
df['user'] = df['User_Id'].map(user_to_user_encoded)

place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id')
df['place'] = df['Place_Id'].map(place_to_place_encoded)
```

**Proses yang Dilakukan:**
Membuat fungsi khusus untuk mengonversi User_Id dan Place_Id menjadi indeks numerik menggunakan dictionary mapping dua arah, lalu menerapkan hasilnya ke dataset.

**Alasan Dilakukan:**
* Model machine learning hanya dapat memproses input numerik, tidak dapat memproses string atau ID kategorikal
* Dictionary encoding memberikan fleksibilitas dalam mapping antara ID asli dan encoded saat diperlukan
* Memastikan tiap user dan place memiliki representasi numerik unik yang berurutan mulai dari 0
* Memberikan kontrol yang lebih fleksibel dibanding LabelEncoder, sesuai kebutuhan sistem rekomendasi
* Encoding manual memudahkan penyesuaian untuk skenario spesifik proyek

#### 6. Normalisasi Rating Manual

**Kode Program:**
```python
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)
min_rating, max_rating = min(df['Place_Ratings']), max(df['Place_Ratings'])

y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```

**Proses yang Dilakukan:**
Mengonversi kolom rating ke tipe float32, lalu menerapkan normalisasi min-max untuk merubah skala nilai rating menjadi rentang 0 hingga 1.

**Alasan Dilakukan:**
* Input dengan rentang terbatas seperti 0-1 membuat pelatihan model neural network atau embedding lebih stabil
* Menghindari bias akibat skala rating yang berbeda-beda yang dapat mempengaruhi pembelajaran model
* Mempercepat proses konvergensi model karena gradien menjadi lebih stabil
* Implementasi manual terhadap normalisasi memungkinkan penyesuaian sesuai kebutuhan spesifik proyek
* Menghindari dominasi fitur akibat perbedaan skala sehingga performa model lebih optimal

#### 7. Random Shuffle Data

**Kode Program:**
```python
df = df.sample(frac=1, random_state=42)
```

**Proses yang Dilakukan:**
Mengacak seluruh baris data menggunakan fungsi `sample` dengan parameter `frac=1` untuk mengambil semua data secara acak dan `random_state=42` agar hasil acakan konsisten (reproducible).

**Alasan Dilakukan:**
* Mengurangi potensi bias yang muncul jika data tersusun berurutan, misalnya berdasarkan waktu atau kategori tertentu
* Memastikan pembagian data training dan validasi nantinya memiliki distribusi yang seimbang dan acak
* Membantu model belajar dengan lebih general karena variasi data yang lebih menyeluruh
* Mencegah model menangkap pola dari urutan data yang sebenarnya tidak relevan terhadap tujuan prediksi

#### 8. Split Data Manual

**Kode Program:**
```python
x = df[['user', 'place']].values
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```

**Proses yang Dilakukan:**
Memisahkan fitur `user` dan `place` sebagai input (X), lalu menghitung indeks pembatas untuk membagi 80% data sebagai training dan 20% sisanya sebagai validasi, menggunakan pembagian manual berbasis indexing.

**Alasan Mengapa Diperlukan:**
- Kontrol presisi terhadap proporsi pembagian data dengan rasio 80:20 yang optimal untuk training dan evaluasi
- Implementasi manual memungkinkan penyesuaian khusus sesuai kebutuhan project dan karakteristik data
- Memastikan pembagian data yang konsisten dan reproducible untuk perbandingan performa antar model
- Memberikan fleksibilitas dalam mengatur strategi pembagian data tanpa tergantung pada library eksternal
- Memungkinkan evaluasi model yang objektif dengan data validasi yang terpisah dari training
- 
### Insight 

Setiap langkah dalam tahap data preparation memiliki peran penting untuk memastikan data yang digunakan dalam modeling sudah bersih, terstruktur, dan sesuai kebutuhan. Pendekatan manual yang diterapkan, seperti pada proses encoding, normalisasi, dan pembagian data, memberikan fleksibilitas penuh agar sistem rekomendasi yang dikembangkan benar-benar sesuai dengan kebutuhan dan tujuan akhir project. Perlu diingat, kualitas data preparation menentukan sekitar 70% dari keberhasilan sebuah project machine learning. Dengan menjalankan tahapan ini secara sistematis dan teliti, kita memastikan bahwa model Content-Based Filtering dan Collaborative Filtering dapat belajar secara optimal dan menghasilkan rekomendasi yang akurat, relevan, serta bernilai bagi pengguna.

#### Catatan Penting:

* Normalisasi dilakukan manual karena model berbasis embedding atau neural network lebih stabil saat menerima input dalam rentang kecil (0–1).
* Proses encoding ID menggunakan dictionary custom untuk mempermudah mapping kembali ke ID asli saat interpretasi hasil.
* Split data dilakukan manual untuk memberikan kontrol penuh atas proporsi dan komposisi data training versus validasi.
* Urutan tahapan sudah mengikuti implementasi aktual sesuai yang ada di notebook, agar selaras antara dokumentasi dan eksekusi teknis.

---

## Modeling

### 1. Content-Based Filtering (CBF)

Content-Based Filtering (CBF) adalah pendekatan yang merekomendasikan item dalam konteks ini, tempat wisata berdasarkan kemiripan karakteristik atau kontennya. Setelah sebelumnya kita mengubah data kategori tempat wisata menjadi representasi vektor numerik dengan metode TF-IDF pada tahap data preparation, sistem CBF dikembangkan melalui langkah-langkah berikut:

#### Pembangunan Sistem CBF dengan Cosine Similarity

Pada sistem ini, **Cosine Similarity** dimanfaatkan untuk mengukur tingkat kemiripan antar tempat wisata dengan membandingkan vektor TF-IDF masing-masing. Semakin tinggi nilai cosine similarity antara dua tempat, semakin besar kemungkinan keduanya dianggap mirip dan direkomendasikan kepada pengguna yang menyukai salah satunya.

#### Rumus Cosine Similarity:

$$
\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}
$$

Keterangan:

* \$A, B\$ → vektor representasi dari dua tempat wisata yang dibandingkan
* \$A \cdot B\$ → hasil perkalian dot product antara kedua vektor
* \$|A|\$ → panjang atau norma dari vektor \$A\$

Dengan menggunakan rumus ini, kita dapat menghitung sejauh mana dua tempat wisata memiliki kesamaan konten. Nilai cosine similarity berkisar antara 0 (tidak mirip sama sekali) hingga 1 (sangat mirip). Semakin mendekati 1, semakin kuat rekomendasinya untuk pengguna yang sudah menyukai salah satu tempat tersebut.

#### Implementasi Sistem CBF:

```python
# Membuat model Content-Based Filtering berbasis kategori tempat wisata dengan menggunakan TF-IDF
tfidf_vectorizer_for_category = TfidfVectorizer()
tfidf_vectorizer_for_category.fit(df_place['Category'])

# Mengubah data kategori tempat wisata menjadi representasi numerik menggunakan TF-IDF dan mengecek dimensinya
tfidf_matrix = tfidf_vectorizer_for_category.fit_transform(df_place['Category'])

# Menghitung matriks kemiripan cosine antar tempat wisata berdasarkan representasi TF-IDF kategori mereka
cosine_sim = cosine_similarity(tfidf_matrix)

#  Membuat DataFrame dari matriks kemiripan cosine dengan nama tempat wisata sebagai indeks dan kolom
cosine_sim_df = pd.DataFrame(
    cosine_sim, index=df_place.Place_Name, columns=df_place.Place_Name)

# Fungsi ini memberikan rekomendasi tempat wisata mirip berdasarkan kemiripan cosine kategori, dengan menerima nama tempat sebagai input dan mengembalikan daftar k tempat wisata terdekat
def destination_recommendations(place_name, similarity_data=cosine_sim_df, 
                               items=df_place[['Place_Name', 'Category']], k=10):
    index = similarity_data.loc[:,place_name].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)
```

Fungsi `destination_recommendations` akan memberikan daftar rekomendasi sebanyak `k` tempat yang paling mirip dengan tempat yang diminta, berdasarkan kemiripan kategori yang dihitung dengan cosine similarity.

#### Hasil Rekomendasi CBF

Jika pengguna menyukai **Museum Gedung Sate**, berikut adalah daftar 10 tempat wisata yang direkomendasikan berdasarkan kemiripan kategori:

| Rank | Place Name                              | Category |
|------|-----------------------------------------|----------|
| 1    |Taman Budaya Jawa Barat                  | Budaya   |
| 2    |Monumen Perjuangan Rakyat Jawa Barat     | Budaya   |
| 3    |Monumen Bandung Lautan Api               | Budaya   |
| 4    |Museum Pos Indonesia                     | Budaya   |
| 5    |Museum Geologi Bandung                   | Budaya   |
| 6    |Museum Pendidikan Nasional               | Budaya   |
| 7    |Museum Sri Baduga                        | Budaya   |
| 8    |Museum Barli                             | Budaya   |
| 9    |Roemah Seni Sarasvati                    | Budaya   |
| 10   |Taman Sejarah Bandung                    | Budaya   |

Semua rekomendasi memiliki kategori “Budaya” karena sistem Content-Based Filtering menghitung kemiripan berdasarkan atribut konten (kategori tempat), sehingga tempat-tempat yang memiliki kesamaan tema akan muncul sebagai rekomendasi utama.

#### Visualisasi Gambar Content Based Filtering

Hasil keluaran dari sistem Content-Based Filtering. Visualisasi ini menunjukkan bagaimana sistem memberikan rekomendasi tempat wisata berdasarkan kemiripan konten (dalam hal ini kategori) dengan tempat yang disukai pengguna.

![Content Based Filtering](https://github.com/user-attachments/assets/13d83b4d-880f-4952-b44e-6c4d634a8afb)

*Gambar 2. Visualisasi Hasil Output dari Content Based Filtering*

### 2. Collaborative Filtering 

Collaborative Filtering adalah teknik rekomendasi yang memanfaatkan pola interaksi antar pengguna dan item, tanpa melihat konten dari item itu sendiri. Metode ini mengandalkan kesamaan perilaku antar pengguna untuk memprediksi preferensi.

Pada proyek ini, pendekatan yang digunakan adalah berbasis deep learning dengan membangun model **RecommenderNet** menggunakan framework TensorFlow.

#### Tahapan Preprocessing Data untuk CF

```python
# Encoding User_Id dan Place_Id
def dict_encoder(col, data=df):
    unique_val = data[col].unique().tolist()
    val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
    return val_to_val_encoded, val_encoded_to_val

# # Encoding User_Id
user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id')

# Encoding Place_Id
place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id')

# Normalisasi rating
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)
min_rating, max_rating = min(df['Place_Ratings']), max(df['Place_Ratings'])
```

#### Struktur Model RecommenderNet

Model RecommenderNet ini dibuat sebagai subclass dari `tf.keras.Model`, dirancang khusus untuk menangani rekomendasi berbasis embedding. 

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_size = embedding_size

        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.places_embedding = layers.Embedding(
            num_places,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.places_bias = layers.Embedding(num_places, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        places_vector = self.places_embedding(inputs[:, 1])
        places_bias = self.places_bias(inputs[:, 1])

        dot_user_places = tf.tensordot(user_vector, places_vector, 2)
        x = dot_user_places + user_bias + places_bias

        return tf.nn.sigmoid(x)
```

#### Konfigurasi Model

Bagian ini digunakan untuk mengonfigurasi model RecommenderNet sebelum pelatihan.

```python
model = RecommenderNet(num_users, num_place, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.0004),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

#### Rumus Estimasi Rating pada Collaborative Filtering

$$
\hat{r}_{u, i} = \sigma(\mathbf{p}_u \cdot \mathbf{q}_i + b_u + b_i)
$$

Keterangan:

* \$\mathbf{p}\_u\$: vektor embedding yang merepresentasikan preferensi pengguna \$u\$
* \$\mathbf{q}\_i\$: vektor embedding yang menggambarkan karakteristik tempat wisata \$i\$
* \$b\_u\$: bias spesifik pengguna, menangkap kecenderungan pengguna memberi rating tinggi atau rendah secara umum
* \$b\_i\$: bias spesifik tempat, menangkap seberapa populer atau tidak populer suatu tempat secara umum
* \$\sigma\$: fungsi sigmoid, digunakan untuk membatasi output akhir ke dalam rentang \[0, 1] sehingga sesuai dengan skala rating ter-normalisasi

#### Hasil Rekomendasi Collaborative Filtering
 
 **Rekomendasi Sistem untuk User 111**

 **Tempat wisata dengan rating tertinggi untuk User 111:**

| Place Name                             | Category       |
|----------------------------------------|----------------|
| Teras Cikapundung BBWS                 | Taman Hiburan  |
| Wisata Batu Kuda                       | Cagar Alam     |
| Museum Gedung Sate                     | Budaya         |
| Masjid Agung Trans Studio Bandung      | Tempat Ibadah  |
| Sanghyang Heuleut                      | Cagar Alam     |

 **Top-10 rekomendasi tempat wisata untuk User 111:**

| No | Place Name                       | Category       | Price | Rating |
|----|----------------------------------|----------------|-------------|----------------|
| 1  | Museum Pos Indonesia            | Budaya         | 0           | 4.5            |
| 2  | Situ Patenggang                 | Cagar Alam     | 20000       | 4.5            |
| 3  | Kawah Rengganis Cibuni          | Cagar Alam     | 5000        | 4.3            |
| 4  | Pemandian Air Panas Cimanggu    | Cagar Alam     | 23000       | 3.9            |
| 5  | Glamping Lakeside Rancabali     | Taman Hiburan  | 30000       | 4.4            |
| 6  | Bukit Jamur                     | Cagar Alam     | 0           | 4.2            |
| 7  | Happyfarm Ciwidey              | Cagar Alam     | 15000       | 4.2            |
| 8  | Taman Miniatur Kereta Api       | Taman Hiburan  | 15000       | 4.4            |
| 9  | Rainbow Garden                  | Cagar Alam     | 20000       | 4.6            |
| 10 | Kota Mini                       | Taman Hiburan  | 20000       | 4.4            |

Analisis singkat berdasarkan rekomendasi untuk User 111:

* Mayoritas rekomendasi (5 dari 10) berasal dari kategori **Cagar Alam**, menunjukkan preferensi pengguna terhadap wisata alam.
* Rata-rata rating rekomendasi cukup tinggi, sekitar **4.45**, yang menandakan sistem memilih tempat yang berkualitas.
* Harga tiket bervariasi dari **gratis sampai Rp 30.000**, memberikan pilihan yang beragam untuk berbagai segmen pengguna.
* Rekomendasi mencakup berbagai jenis tempat, sehingga menawarkan variasi bagi pengguna.

#### Visualisasi Gambar Collaborative Filtering

Gambar ini menggambarkan hasil output dari sistem Collaborative Filtering.

![Collaborative Filtering](https://github.com/user-attachments/assets/2c808468-689f-408f-9b03-ba1017633181)

*Gambar 3. Visualisasi Hasil Output dari Collaborative Filtering*

---

## Evaluation

### Evaluasi Content-Based Filtering (CBF)

Untuk pendekatan CBF, evaluasi dilakukan dengan menggunakan metrik yang menilai relevansi hasil rekomendasi. Karena metode ini tidak memanfaatkan rating eksplisit dari pengguna, penilaian difokuskan pada seberapa relevan rekomendasi yang dihasilkan dengan kategori dan konten tempat wisata.

#### Metrik Evaluasi yang Digunakan

##### 1. Precision@K

Precision@K mengukur seberapa besar proporsi item yang relevan dari total K rekomendasi teratas.

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan dalam top-K}}{\text{K}}
$$

#### Implementasi Evaluasi CBF

```python
def evaluate_cbf_precision_at_k(place_name, k=10):
    # Mendapatkan kategori tempat input
    input_category = df_place[df_place['Place_Name'] == place_name]['Category'].iloc[0]

    # Mendapatkan rekomendasi
    recommendations = destination_recommendations(place_name, k=k)

    # Menghitung item yang relevan (kategori sama)
    relevant_count = sum(1 for cat in recommendations['Category'] if cat == input_category)

    precision = relevant_count / k
    return precision, relevant_count
```

#### Hasil Evaluasi CBF

Evaluasi dilakukan untuk tempat wisata **Museum Gedung Sate** (kategori: Budaya), dan diperoleh hasil berikut:

| Metrik | K=5 | K=10 |
|--------|-----|------|
| Precision@K | 1.00 | 1.00 |
| Relevant Items | 5/5 | 10/10 |

#### Analisis Hasil CBF

* **Precision\@5 = 1.00** menunjukkan semua dari 5 rekomendasi teratas berada dalam kategori yang sama, yaitu Budaya.
* **Precision\@10 = 1.00** menunjukkan seluruh 10 rekomendasi juga relevan secara kategori.
* Ini membuktikan bahwa sistem CBF bekerja dengan sangat baik dalam menghasilkan rekomendasi yang konsisten dan sesuai berdasarkan konten kategori tempat wisata.

#### Evaluasi Kualitas Rekomendasi CBF

Analisis lanjutan untuk rekomendasi **Museum Gedung Sate**:

| Aspek Evaluasi        | Skor  | Keterangan                                                   |
| --------------------- | ----- | ------------------------------------------------------------ |
| Konsistensi Kategori  | 10/10 | Semua rekomendasi berasal dari kategori Budaya               |
| Keragaman (Diversity) | 8/10  | Ada variasi yang cukup dalam jenis tempat Budaya             |
| Relevansi             | 9/10  | Sangat cocok untuk pengguna yang menyukai sejarah dan budaya |

#### Evaluasi Precision

Grafik di bawah ini menampilkan hasil nilai precision yang diperoleh dari data evaluasi:

![nilai_precision](https://github.com/user-attachments/assets/c3d1aef2-21fb-4dd6-a4a1-1eac8ce2ed23)

*Gambar 3. Visualisasi Hasil Precision Evaluasi Content-Based Filtering (CBF)*

### Evaluasi Collaborative Filtering

Pada pendekatan Collaborative Filtering, sistem memanfaatkan model neural network **RecommenderNet** untuk memprediksi rating pengguna terhadap tempat wisata. Evaluasi performa dilakukan menggunakan metrik kuantitatif berbasis akurasi prediksi rating.

#### Evaluasi Kuantitatif dengan RMSE

Root Mean Squared Error (RMSE) digunakan sebagai metrik utama untuk menilai seberapa akurat model dalam memprediksi rating. RMSE menghitung rata-rata selisih kuadrat antara nilai prediksi dan nilai aktual, lalu diakarkan untuk mengembalikan skala aslinya.

$$
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

Keterangan:

* \$y\_i\$: rating aktual (setelah normalisasi)
* \$\hat{y}\_i\$: rating hasil prediksi
* \$n\$: jumlah total sampel evaluasi

#### Callback untuk Evaluasi Otomatis

```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_root_mean_squared_error') < 0.35):
            print('Lapor! Metrik validasi sudah sesuai harapan')
            self.model.stop_training = True
```

#### Hasil Evaluasi Collaborative Filtering

Setelah proses pelatihan selama 100 epoch dengan pemantauan callback:

| Metrik           | Nilai Akhir             | Target     | Status               |
| ---------------- | ----------------------- | ---------- | -------------------- |
| Validation RMSE  | 0.3568                  | < 0.35     | Belum tercapai       |
| Training RMSE    | 0.3127                  | -          | Konvergen            |
| Training Loss    | 0.6493                  | -          | Menurun              |
| Validation Loss  | 0.7248                  | -          | Stabil               |
| Status Pelatihan | Selesai (100/100 epoch) | Early Stop | Callback tidak aktif |

**Analisis Performa Collaborative Filtering:**

* Model menunjukkan tanda-tanda **overfitting**, yang terlihat dari perbedaan cukup besar antara nilai metrik pada data training dan validasi.
* Nilai validation RMSE terbaik tercatat pada epoch ke-22 dan ke-24 sebesar 0.3512, namun setelah itu mengalami fluktuasi tanpa perbaikan signifikan.
* Training loss mengalami penurunan dari 0.7208 menjadi 0.6493, sedangkan validation loss tetap relatif stabil di kisaran 0.72.

Walaupun target RMSE kurang dari 0.35 belum tercapai (tersisa selisih 0.0068), model masih mampu menghasilkan prediksi yang cukup baik, namun perlu upaya lebih lanjut untuk meningkatkan kemampuan generalisasi melalui penerapan teknik regularisasi dan penggunaan early stopping.

#### Visualisasi Grafik RMSE Validasi

Grafik berikut menggambarkan perubahan nilai RMSE pada data validasi sepanjang proses pelatihan:

![grafik_RMSE](https://github.com/user-attachments/assets/e5d3ac72-320c-4c6a-8323-ad42c0298d54)

*Gambar 4. Visualisasi Grafik RMSE Validasi*

#### Analisis Kualitas Rekomendasi Collaborative Filtering

Evaluasi kualitas rekomendasi untuk User 111:

| Aspek Evaluasi      | Analisis                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------- |
| **Personalisasi**   | Rekomendasi bervariasi dan disesuaikan dengan riwayat preferensi pengguna                |
| **Keragaman**       | Mencakup 4 kategori berbeda: Taman Hiburan, Cagar Alam, Budaya, Tempat Ibadah            |
| **Kualitas Rating** | Rata-rata rating tinggi sebesar 4.45 (dengan rentang antara 4.2–4.6)                     |
| **Rentang Harga**   | Pilihan harga beragam, mulai dari gratis hingga Rp30.000                                 |
| **Relevansi**       | Selaras dengan pola rating historis pengguna, menunjukkan rekomendasi yang tepat sasaran |

#### Perbandingan dengan Preferensi Historis

User 111 memiliki riwayat preferensi:

* **Taman Hiburan** (Teras Cikapundung BBWS)
* **Cagar Alam** (Wisata Batu Kuda, Sanghyang Heuleut)
* **Budaya** (Museum Gedung Sate)
* **Tempat Ibadah** (Masjid Agung Trans Studio Bandung)

Rekomendasi yang diberikan sistem mencerminkan preferensi ini dengan distribusi sebagai berikut:

* 10% Taman Hiburan (1 dari 10 rekomendasi)
* 20% Cagar Alam (2 dari 10 rekomendasi)
* 10% Budaya (1 dari 10 rekomendasi)
* 10% Tempat Ibadah (1 dari 10 rekomendasi)

---
Berikut kalimat tabel yang sudah diperbaiki dan dirapikan:

---

### Perbandingan Performa Model

| Model   | Kelebihan                                                                                         | Kekurangan                                                                                                                              | Skor Performa |
| ------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| **CBF** | - Precision kategori: 100%<br>- Mendukung cold start<br>- Konsisten dan mudah diprediksi          | - Terbatas pada metadata<br>- Kurang variasi<br>- Tidak dipersonalisasi secara spesifik                                                 | 8.5/10        |
| **CF**  | - Sangat terpersonalisasi<br>- Rekomendasi lebih beragam<br>- Rata-rata rating tinggi (avg: 4.45) | - Membutuhkan data historis pengguna<br>- Masalah cold start<br>- RMSE validasi belum capai target (final epoch: 0.3568, target < 0.35) | 8.4/10        |

---

## Analisis Performa dan Hasil Modeling

### 1. Performa Model

Dua pendekatan utama dikembangkan dalam sistem rekomendasi ini:

* **Content-Based Filtering (CBF)** menunjukkan performa sangat baik dengan precision sempurna (100%) dalam menyarankan destinasi sesuai kategori. Model ini menghasilkan rekomendasi yang konsisten, terutama efektif untuk pengguna baru atau pengguna tanpa riwayat interaksi sebelumnya.
* **Collaborative Filtering (CF)** mampu memberikan prediksi yang cukup akurat dengan nilai RMSE validasi **0.3568**, sedikit melampaui target 0.35. Meski target metrik belum sepenuhnya tercapai, performa model tetap kompetitif, menghasilkan rekomendasi yang beragam dan personal.

Selama proses training selama 100 epoch, model CF menunjukkan tanda-tanda **overfitting**, terlihat dari selisih mencolok antara training loss yang terus menurun dan validation loss yang relatif stagnan. Ini mengindikasikan perlunya perbaikan, misalnya dengan teknik regularisasi atau penyesuaian strategi early stopping.

### 2. Karakteristik Rekomendasi

* **CBF:** Menghasilkan rekomendasi yang homogen dan sangat relevan (contoh: seluruh tempat wisata dari kategori Budaya). Cocok untuk pengguna dengan preferensi spesifik terhadap kategori tertentu.
* **CF:** Menyediakan rekomendasi yang lebih bervariasi, mencakup berbagai kategori (Taman Hiburan, Budaya, Cagar Alam, Tempat Ibadah), serta mampu menangkap pola preferensi pengguna berdasarkan histori interaksi.

### 3. Kualitas Output

* **CBF:** Fokus pada konsistensi dan relevansi, ideal untuk skenario cold start, dengan hasil yang stabil dan dapat diprediksi.
* **CF:** Menyajikan keseimbangan antara **personalization**, **diversity**, dan **rating kualitas konten** dengan rata-rata rating rekomendasi mencapai 4.45 serta distribusi harga tiket yang bervariasi.

### 4. Rekomendasi Model

* Untuk **pengguna baru** atau kasus tanpa data interaksi, **CBF** menjadi pilihan utama karena tidak memerlukan data historis.
* Untuk **pengguna aktif**, **CF** lebih unggul karena mampu menghadirkan rekomendasi yang dipersonalisasi sesuai pola perilaku dan preferensi pengguna.

---

## Kesimpulan

1. Sistem rekomendasi wisata berbasis machine learning berhasil dikembangkan dengan memadukan dua pendekatan yang saling melengkapi: Content-Based Filtering (CBF) dan Collaborative Filtering (CF).

2. **Hasil evaluasi menunjukkan performa yang beragam**:

   * **CBF** berhasil mencapai Precision\@10 sebesar 100%, menunjukkan konsistensi sangat baik dalam menyarankan destinasi berdasarkan kategori.
   * **CF** meskipun belum mencapai target RMSE < 0.35 (dengan capaian 0.3568), tetap mampu menghasilkan rekomendasi yang bersifat personal dan variatif.

3. **Ciri khas masing-masing model**:

   * **CBF** unggul dalam hal **konsistensi** terhadap preferensi berbasis kategori.
   * **CF** menawarkan **personalisasi** yang kuat dengan keberagaman rekomendasi, meskipun masih ada ruang untuk peningkatan performa (terutama pada metrik RMSE).

4. Proyek ini berhasil menjawab tantangan bisnis dengan menyediakan rekomendasi wisata yang akurat, relevan, dan sesuai kebutuhan pengguna di area Bandung.

5. Perpaduan kedua pendekatan memberikan solusi yang menyeluruh, mampu memenuhi kebutuhan baik pengguna baru maupun pengguna aktif yang sering berinteraksi.

6. Sistem ini memiliki potensi besar untuk dikembangkan lebih lanjut ke tahap implementasi komersial, mengingat hasil evaluasi yang mendekati target serta kualitas rekomendasi yang sudah tergolong tinggi.

---

## Referensi

* Yanuar, R., & Sabani, W. A. P. (2020). Aplikasi Pembuat Jadwal Kunjungan Wisata untuk Wisatawan Lokal di Kota Bandung Berbasis Android. *eProceedings of Applied Science*, 6(3).

* Pasaribu, Y. S., & Sitompul, T. S. (2023). Rekomendasi Destinasi Wisata Kota Bandung Menggunakan Algoritma Collaborative Filtering. *Mutiara: Jurnal Penelitian dan Karya Ilmiah*, 1(6), 382–392. [https://doi.org/10.59059/mutiara.v1i6.736](https://doi.org/10.59059/mutiara.v1i6.736)

* Adlan, M. N., & Setiawan, E. B. (2025). Sistem Rekomendasi Destinasi Wisata di Kota Bandung dengan Collaborative Filtering Menggunakan K-Nearest Neighbors. *eProceedings of Engineering*, 12(1), 2129–2136.
