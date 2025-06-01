# Movies Recommendation System Report

---

# **Project Overview**

Dengan terus bertambahnya jumlah film yang tersedia di berbagai platform digital, pengguna sering kesulitan menemukan film yang sesuai dengan selera mereka. Salah satu pendekatan populer untuk mengatasi permasalahan ini adalah sistem rekomendasi berbasis konten (*content-based filtering*), di mana rekomendasi diberikan berdasarkan kesamaan karakteristik konten — dalam hal ini, genre film.

Proyek ini bertujuan membangun **sistem rekomendasi film berdasarkan genre** menggunakan pendekatan content-based filtering. Misalnya, jika seorang pengguna menyukai film *King Kong*, maka sistem akan merekomendasikan film lain yang memiliki genre serupa seperti *Action*, *Adventure*, atau *Horror*.

Sistem ini relevan karena membantu pengguna mengeksplorasi film yang mungkin belum pernah mereka tonton namun memiliki karakteristik yang mereka sukai. Selain itu, pendekatan ini tidak memerlukan data interaksi pengguna lain, sehingga cocok untuk kondisi *cold start* atau sistem baru.

---

# **Business Understanding**

## **Problem Statements**

1. Bagaimana cara merepresentasikan informasi genre film secara numerik agar bisa digunakan dalam perhitungan kemiripan antar film?
2. Bagaimana cara mengukur tingkat kemiripan antar film hanya berdasarkan informasi kontennya, khususnya genre?
3. Bagaimana cara mengembangkan sistem rekomendasi film yang mampu memberikan saran film sejenis hanya dari satu input judul film?

## **Goals**

1. Menghasilkan sistem rekomendasi film yang mampu menyarankan film lain dengan genre yang mirip dari input satu judul film.
2. Memudahkan pengguna menemukan film-film baru sesuai dengan preferensi genre mereka tanpa perlu memberikan penilaian eksplisit.
3. Membuat sistem yang bersifat general dan bisa digunakan tanpa ketergantungan pada data pengguna (*user rating history*).

## **Solution Statements**

1. Menggunakan pendekatan content-based filtering dengan teknik TF-IDF vectorization untuk merepresentasikan genre film.
2. Mengukur kemiripan antar film menggunakan cosine similarity untuk menentukan film-film yang memiliki kemiripan konten.
3. Mengimplementasikan fungsi rekomendasi yang menerima judul film sebagai input dan menghasilkan daftar rekomendasi berdasarkan tingkat kemiripan genre.

---

# **Data Understanding**

Tahap *Data Understanding* bertujuan untuk mengenal struktur, isi, dan karakteristik data yang akan digunakan. Dalam proyek ini, digunakan dua dataset utama dari [Kaggle - The Movies Dataset](https://www.kaggle.com/api/v1/datasets/download/rounakbanik/the-movies-dataset), yaitu `ratings.csv` dan `movies_metadata.csv`. Proses understanding mencakup membaca data, memeriksa struktur kolom dan tipe datanya, serta melakukan eksplorasi awal seperti statistik deskriptif untuk mengidentifikasi pola umum, missing values, dan anomali. 

## **Dataset: Ratings**

```python
data_ratings = pd.read_csv("/content/ratings.csv")
data_ratings.info()
````
RangeIndex: 26024289 entries, 0 to 26024288

Data columns (total 4 columns):

| Kolom     | Deskripsi                                                           |
| --------- | ------------------------------------------------------------------- |
| userId    | ID pengguna yang memberikan rating.                                 |
| movieId   | ID film yang diberi rating.                                         |
| rating    | Nilai rating yang diberikan oleh pengguna (biasanya skala 1.0–5.0). |
| timestamp | Waktu pemberian rating dalam format Unix timestamp.                 |

**Insight:**

Data ratings memiliki 26.024.289 baris dan 4 fitur.

## **Dataset: Movies Metadata**

```python
data_movies = pd.read_csv("/content/movies_metadata.csv")
data_movies.info()
```

RangeIndex: 45466 entries, 0 to 45465

Data columns (total 24 columns):
| Kolom                   | Deskripsi                                     |
| ----------------------- | --------------------------------------------- |
| adult                   | Apakah film untuk dewasa (True/False).        |
| belongs\_to\_collection | Informasi jika film bagian dari seri/koleksi. |
| budget                  | Anggaran produksi film.                       |
| genres                  | Daftar genre film.                            |
| homepage                | URL resmi film (jika tersedia).               |
| id                      | ID unik film dari TMDB.                       |
| imdb\_id                | ID film di IMDb.                              |
| original\_language      | Bahasa asli film.                             |
| original\_title         | Judul asli film.                              |
| overview                | Sinopsis film.                                |
| popularity              | Skor popularitas dari TMDB.                   |
| poster\_path            | Path ke poster film.                          |
| production\_companies   | Perusahaan produksi film.                     |
| production\_countries   | Negara tempat film diproduksi.                |
| release\_date           | Tanggal rilis.                                |
| revenue                 | Pendapatan film.                              |
| runtime                 | Durasi film.                                  |
| spoken\_languages       | Bahasa yang digunakan dalam film.             |
| status                  | Status rilis film.                            |
| tagline                 | Slogan/tagline film.                          |
| title                   | Judul film (untuk ditampilkan).               |
| video                   | Apakah berupa video (True/False).             |
| vote\_average           | Rata-rata rating pengguna.                    |
| vote\_count             | Jumlah total suara/rating.                    |

**Insight:**

Data movies memiliki 45.466 baris dan 24 kolom.

---

### **Deskripsi Statistik Ratings**

```python
ratings.describe()
```

| Kolom     | Insight                                                               |
| --------- | --------------------------------------------------------------------- |
| userId    | Rentang luas (8 – 270.761), variatif, artinya banyak user berbeda.    |
| movieId   | Rentang sangat luas, distribusi tidak merata (skewed).                |
| rating    | Rata-rata 3.53 (rentang 0.5 – 5), mayoritas rating cenderung positif. |
| timestamp | Rentang waktu luas, distribusi data pengambilan berlangsung lama.     |

---

### **Deskripsi Statistik Movies**

```python
movies.describe()
```

| Kolom         | Insight                                                                                                                                |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| revenue       | Rata-rata 10.7 juta, tapi sangat bervariasi. 75% film tidak mencatatkan revenue (nol), kemungkinan besar data tidak lengkap.           |
| runtime       | Rata-rata durasi 94 menit. Banyak film berdurasi antara 85–107 menit. Nilai ekstrim bisa jadi karena data tidak valid.                 |
| vote\_average | Rata-rata 5.61, mayoritas film mendapat rating 5–7. Nilai minimum 0 bisa berarti belum ada rating masuk.                               |
| vote\_count   | Rata-rata 107, tapi distribusi sangat skewed. Banyak film dengan vote sedikit, hanya sedikit film yang sangat populer (hingga 12.269). |

## **3. Exploratory Data Analysis**

Exploratory Data Analysis (EDA) dilakukan untuk memahami pola, anomali, serta hubungan antar fitur dalam dataset sebelum masuk ke tahap pemodelan. Dalam proses ini, dilakukan visualisasi distribusi data seperti rating film, tahun rilis, durasi, hingga korelasi antar fitur numerik. EDA juga membantu dalam pengambilan keputusan terhadap data mana yang perlu dibersihkan, difilter, atau dipertahankan.

### **a. Distribusi Rating**

<p align="center">
  <img src="images/3a.png"width="500"/>
</p>

**Insight :**

Grafik tersebut menunjukkan distribusi rating film dalam dataset, dengan tambahan garis KDE (Kernel Density Estimation) yang menggambarkan bentuk distribusi secara halus. Terlihat bahwa sebagian besar rating berkumpul di sekitar angka 3 hingga 4, yang berarti mayoritas pengguna memberikan penilaian cukup positif terhadap film. Puncak-puncak tajam pada garis KDE menandakan adanya frekuensi tinggi pada nilai rating tertentu, seperti 3.0, 4.0, dan 5.0. Hal ini bisa disebabkan oleh kebiasaan pengguna memberikan rating bulat. Sementara itu, rating di bawah 2 terlihat jauh lebih jarang, mengindikasikan bahwa film dengan penilaian sangat rendah lebih sedikit jumlahnya atau jarang diberikan oleh pengguna. Grafik ini menunjukkan bahwa persebaran rating cenderung positif dengan sedikit outlier di sisi rating rendah.

---

### **b. Distribusi Tahun Rilis Film**

<p align="center">
  <img src="images/3b.png"width="500"/>
</p>

**Insight :**

Grafik tersebut menunjukkan distribusi jumlah film berdasarkan tahun rilisnya. Terlihat bahwa produksi film masih sangat sedikit sebelum tahun 1950, lalu mulai mengalami peningkatan secara bertahap setelahnya. Lonjakan yang signifikan terjadi sejak era 1990-an hingga mencapai puncaknya sekitar tahun 2010-an. Hal ini mencerminkan perkembangan industri perfilman global, baik dari sisi teknologi, permintaan pasar, maupun kemudahan produksi dan distribusi. Penurunan setelah tahun 2015 kemungkinan disebabkan oleh data yang belum lengkap atau keterbatasan koleksi dataset, bukan berarti produksi film benar-benar menurun secara drastis. Grafik ini menegaskan bahwa mayoritas film dalam dataset berasal dari dua dekade terakhir.

---

### **c. Distribusi Runtime**

<p align="center">
  <img src="images/3c.png"width="500"/>
</p>

**Insight :**

1. Film paling umum berdurasi 90–100 menit.

2. Ada sejumlah kecil film yang sangat pendek atau panjang, tapi tidak mendominasi.

3. Durasi 0 kemungkinan perlu diperlakukan sebagai data tidak valid dalam analisis lanjutan.

---

### **d. Rating Rata-Rata per Film**

<p align="center">
  <img src="images/3d.png"width="500"/>
</p>

**Insight :**

1. **Distribusi Mendekati Normal**
   - Sebagian besar film memiliki rating rata-rata **2.5–4.0**.

2. **Pola "Bergelombang"**
   - Lonjakan di angka bulat seperti **3.0**, **4.0**, dan **5.0**.

3. **Rating Sangat Rendah Jarang**
   - Film dengan rating **1.0–2.0** sangat sedikit.

4. **Ada Film dengan Rating Mendekati 5.0**
   - Beberapa film sangat disukai (cult classic atau masterpiece).

5. **Skewness Tidak Terlalu Ekstrem**
   - Distribusi relatif simetris.

---

### **e. Korelasi Antar Fitur Numerik Film**

<p align="center">
  <img src="images/3e.png"width="500"/>
</p>

**Insight :**

* Semua fitur numerik memiliki **korelasi rendah (|r| < 0.2)**.
* Artinya: **tidak ada hubungan linear yang kuat** antar fitur numerik.
* Ini menunjukkan bahwa:

  * Popularitas (jumlah rating) **tidak selalu beriringan** dengan kualitas (rating rata-rata).
  * Durasi film **bukan indikator utama** apakah film disukai atau tidak.
  * Tahun rilis juga **bukan penentu utama** nilai atau popularitas film.

Tabel Interpretasi Korelasi

| Fitur 1        | Fitur 2          | Korelasi | Interpretasi                                                                                                      |
| -------------- | ---------------- | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `runtime`      | `release_year`   | 0.09     | Korelasi sangat lemah dan positif — film modern cenderung sedikit lebih panjang durasinya, tapi tidak signifikan. |
| `runtime`      | `average_rating` | -0.06    | Korelasi sangat lemah dan negatif — durasi film **tidak berpengaruh signifikan** terhadap rating.                 |
| `release_year` | `average_rating` | 0.02     | Hampir tidak ada hubungan — film lama atau baru punya peluang yang sama untuk disukai.                            |
| `rating_count` | `average_rating` | 0.04     | Hampir tidak ada korelasi — banyaknya orang yang memberi rating **tidak menjamin film disukai**.                  |
| `runtime`      | `rating_count`   | 0.02     | Tidak ada hubungan berarti antara panjang film dan seberapa banyak orang menontonnya.                             |

---

### **f. Top 10 Movies Berdasarkan Popularity**

<p align="center">
  <img src="images/3f.png"width="500"/>
</p>

**Insight :**

1. **Minions** menjadi film **paling populer secara mencolok**, dengan selisih yang sangat besar dibandingkan film lainnya. Ini menunjukkan keberhasilan luar biasa dalam menjangkau audiens, kemungkinan karena:

   * Segmentasi keluarga dan anak-anak.
   * Visual animasi yang menarik.
   * Strategi pemasaran besar-besaran secara global.

2. **Wonder Woman** dan **Beauty and the Beast** menyusul di peringkat ke-2 dan ke-3 dengan popularitas yang cukup tinggi, namun **masih jauh di bawah Minions**. Ini menunjukkan minat besar pada genre **superhero dan fantasi-musikal**.

3. Beberapa film dalam daftar didominasi oleh genre:

   * **Animasi dan keluarga**: `Minions`, `Big Hero 6`.
   * **Aksi dan petualangan**: `Wonder Woman`, `John Wick`, `Baby Driver`, `Deadpool`, `Avatar`.
   * **Drama dan misteri**: `Gone Girl`.

4. Film **Gone Girl** menempati posisi ke-10, dengan popularitas yang lebih rendah dari yang lain, namun tetap masuk 10 besar — menunjukkan bahwa genre thriller psikologis juga punya daya tarik audiens.

* **Popularitas ≠ Rating Tertinggi**
  Popularitas menggambarkan seberapa banyak film ditonton atau dicari, bukan seberapa baik kualitasnya secara kritis.
  Contoh: *Minions* sangat populer, tapi belum tentu mendapatkan skor tinggi dari kritikus.

* **Faktor genre dan demografi** sangat mempengaruhi popularitas:
  Film yang ramah keluarga atau penuh aksi umumnya menjangkau lebih banyak penonton lintas usia dan budaya.

---

## **4. Data Preparation**

Sebelum dilakukan analisis atau pemodelan lebih lanjut, data perlu dibersihkan dan disiapkan. Tahapan ini meliputi menentukan sampel yang digunakan, konversi format data, merge antar tabel, mengatasi missing value, outlier, menyiapkan fitur agar siap dianalisis dan TF-IDF Vectorizer.

Karena ukuran dataset cukup besar, dilakukan *sampling* sebanyak 10.000 baris untuk masing-masing file guna mempercepat proses eksplorasi dan pengolahan selanjutnya.


### **a. Menentukan sample yang digunakan**

Dikarenakan jumlah data yang terlalu banyak, maka data yang diambil hanya 10.000 data.

```
ratings = data_ratings.sample(n=10000, random_state=42)
ratings.shape
```
> Output : (1000, 4)
**Insight :**

Data ratings berhasil di filter sebanyak 10.000 data.

```
movies = data_movies.sample(n=10000, random_state=42)
movies.shape
```

> Output : (1000, 25)

**Insight :**

Data movies berhasil di filter sebanyak 10.000 data.

### **b. Mengubah Dictionary menjadi String**

Kode ini bertujuan untuk memilih kolom-kolom penting dari DataFrame `movies`, memformat data pada kolom `genres`, dan menyusun ulang data dalam format yang lebih bersih dan terstruktur untuk keperluan analisis lebih lanjut.

---

**Penjelasan Setiap Langkah**

1. **Fungsi `format_genres`**
   Fungsi ini mengubah data dalam kolom `genres` yang awalnya berbentuk string dari list of dictionary (misalnya: `"[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"`) menjadi string biasa yang berisi nama-nama genre yang dipisahkan dengan tanda `|`, seperti:
   `"Action | Adventure"`.

2. **`df = movies[['id', 'title']].copy()`**
   Baris ini menyalin kolom `id` dan `title` dari DataFrame `movies`. Hanya kolom-kolom yang dianggap relevan yang diambil untuk diproses lebih lanjut.

3. **`df['genres'] = movies['genres'].apply(format_genres)`**
   Fungsi `format_genres` diterapkan ke kolom `genres` dalam DataFrame `movies`, kemudian hasilnya disimpan dalam kolom baru `genres` pada DataFrame `df`.

4. **Penyusunan Ulang dan Penggantian Nama Kolom**
   Kolom disusun ulang menjadi urutan `id`, `title`, dan `genres`, kemudian kolom `id` diubah namanya menjadi `movieId` untuk konsistensi dengan format umum pada sistem rekomendasi.

```
python
# Fungsi untuk memformat kolom 'genres' dari string menjadi format string genre yang dipisahkan tanda '|'
def format_genres(genre_string):
    # Tangani nilai NaN atau tipe data non-string
    if pd.isna(genre_string) or not isinstance(genre_string, str):
        return ""

    try:
        # Ubah string literal menjadi list of dict menggunakan ast.literal_eval
        parsed_genres = ast.literal_eval(genre_string)
    except (ValueError, SyntaxError) as e:
        # Jika parsing gagal, tampilkan peringatan dan kembalikan string kosong
        print(f"Warning: Tidak dapat parsing genre string '{genre_string}'. Error: {e}")
        return ""

    # Ambil nilai 'name' dari setiap dictionary dalam list dan gabungkan dengan delimiter '|'
    genre_names = [genre['name'] for genre in parsed_genres if 'name' in genre]
    return " | ".join(genre_names)

# Pilih kolom yang relevan dari DataFrame 'movies'
df = movies[['id', 'title']].copy()

# Terapkan fungsi format_genres ke kolom 'genres' dan simpan ke kolom baru
df['genres'] = movies['genres'].apply(format_genres)

# Susun ulang kolom dan ubah nama kolom 'id' menjadi 'movieId'
df = df[['id', 'title', 'genres']].rename(columns={'id': 'movieId'})

# Tampilkan 5 data teratas sebagai sampel
df.head()
````
**Output :**
```
	movieId	title			genres
43526	411405	Small Crimes		Drama | Comedy | Thriller | Crime
6383	42492	Up the Sandbox		Drama | Comedy
3154	12143	Bad Lieutenant		Crime | Drama
10146	9976	Satan's Little Helper	Horror | Romance | Comedy
9531	46761	Sitcom			Comedy | Drama | Thriller
```

**Insight :**
Data berhasil diubah dari format string kompleks menjadi format yang lebih sederhana dan mudah dipakai, dan disimpan dalam dataframe `df`.

---

### **c. Mengubah tipe data `movieId` menjadi string**

```python
df['movieId'] = df['movieId'].astype(str)
ratings['movieId'] = ratings['movieId'].astype(str)

df.info()
ratings.info()
```
**Hasil Output :**

```
<class 'pandas.core.frame.DataFrame'>
Index: 10000 entries, 43526 to 27147
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   movieId  10000 non-null  object
 1   title    9999 non-null   object
 2   genres   10000 non-null  object
dtypes: object(3)
memory usage: 312.5+ KB
<class 'pandas.core.frame.DataFrame'>
Index: 10000 entries, 11928665 to 13549562
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   userId     10000 non-null  int64  
 1   movieId    10000 non-null  object 
 2   rating     10000 non-null  float64
 3   timestamp  10000 non-null  int64  
dtypes: float64(1), int64(2), object(1)
memory usage: 390.6+ KB
```

**Insight :**
Kolom `movieId` pada dataframe `df` dan `ratings` diubah menjadi string agar konsisten saat digabungkan.

---

### **c. Merge Data**

```python
movie_merged = pd.merge(ratings, df, on='movieId', how="inner")
movie_merged
```

**Hasil Output :**

```
	userId	movieId	rating	timestamp	title							genres
0	43059	1367	4.0	1088197740	Rocky II						Drama
1	255724	4975	1.0	1082336474	Love Is the Devil: Study for a Portrait of Fra...	TV Movie | Drama
2	246259	587	4.0	945228093	Big Fish						Adventure | Fantasy | Drama
3	72228	3022	3.5	1117138601	Dr. Jekyll and Mr. Hyde					Drama | Horror | Science Fiction
4	218871	5991	3.5	1287676841	The Last Laugh						Drama
...	...	...	...	...	...	...
987	116964	2734	3.0	1008741797	David							Drama | History
988	152250	69928	4.0	1339373959	The Man Who Loved Women					Comedy
989	118327	2108	4.0	1017621443	The Breakfast Club					Comedy | Drama
990	183641	919	4.0	1454248623	Blood: The Last Vampire					Fantasy | Animation | Horror | Comedy | Thrill...
991	135992	316	3.0	847883933	Grill Point						Comedy | Drama
992 rows × 6 columns
```

**Insight :**
Data berhasil di-merge berdasarkan kolom `movieId` pada df `ratings` dan `df`, menghasilkan 992 baris dan 6 kolom: `userId`, `movieId`, `rating`, `timestamp`, `title`, dan `genres`.

---

### **d. Handle Missing Values**

```python
movie_merged.isnull().sum()
```

| Kolom      | Jumlah Null |
|------------|-------------|
| userId     | 0           |
| movieId    | 0           |
| rating     | 0           |
| timestamp  | 0           |
| title      | 0           |
| genres     | 0           |

**Insight :**
Tidak terdapat missing values dalam data `movie_merged`.

---

### **e. Handle Duplicated Data**

```python
movie_merged.duplicated().sum()
```

**Hasil Output :**
np.int64(0)

**Insight :**
Tidak terdapat data duplikat dalam `movie_merged`.

---

### f. Menangani Outlier dengan IQR Method
```
# Mengganti Nilai Outlier dengan Batas Atas dan Batas Bawah Data

for i in movie_merged.select_dtypes(include='number'):
    Q1 = movie_merged[i].quantile(0.25)
    Q3 = movie_merged[i].quantile(0.75)
    IQR = Q3 - Q1

    maximum = Q3 + (1.5 * IQR)
    minimum = Q1 - (1.5 * IQR)

    movie_merged[i] = movie_merged[i].mask(movie_merged[i] > maximum, maximum)
    movie_merged[i] = movie_merged[i].mask(movie_merged[i] < minimum, minimum)
```

<p align="center">
  <img src="images/outlier.png"width="500"/>
</p>

**Insight :**

| Fitur         | Insight                                                                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **userId**    | Distribusi user cukup merata tanpa outlier. Semua nilai userId berada dalam rentang wajar (0–270k). Tidak ada titik ekstrem.                                         |
| **rating**    | Rentang rating hanya dari 0.5 hingga 5, dan sudah **tidak ada outlier**. Distribusi simetris dengan median sekitar 3.5. Artinya sistem rating berjalan sesuai skala. |
| **timestamp** | Distribusi waktu rating (dalam format Unix timestamp) juga sudah **tanpa outlier**. Data berkisar dari sekitar 2007 hingga 2017.                                     |

**Kesimpulan :**

* Penanganan outlier berhasil — tidak ada lagi nilai ekstrem atau titik individu terisolasi pada semua fitur numerik.
* Data siap digunakan untuk analisis atau modeling lebih lanjut karena sudah bersih dan stabil.

---

### **g. Memecah Genres yang Dipisahkan oleh `|`**

```python
genres = set()
for genres_str in movie_merged['genres']:
    if genres_str:  # Only process non-empty strings
        for genre in genres_str.split(' | '):
            genres.add(genre.strip())

genres_list = list(genres)
print(genres_list)
```

**Insight :**
Kode ini mengekstrak semua genre unik dari kolom `genres` pada `movie_merged`, memisahkan dengan delimiter `|`, dan menyimpannya dalam list tanpa duplikat. Hasil akhirnya adalah daftar genre unik yang bisa digunakan untuk analisis lanjutan.

### **h. TF-IDF Vectorizer**

Pertama-tama, informasi genre dari setiap film diubah menjadi representasi numerik menggunakan **TF-IDF (Term Frequency - Inverse Document Frequency)**. TF-IDF adalah teknik yang sering digunakan dalam pemrosesan teks untuk merepresentasikan seberapa penting suatu kata (dalam hal ini, genre) dalam suatu dokumen (film) dibandingkan dengan seluruh kumpulan dokumen (film-film lain).

* **Term Frequency (TF)** mengukur seberapa sering suatu genre muncul dalam data film tersebut.
* **Inverse Document Frequency (IDF)** menurunkan bobot genre yang terlalu umum (misalnya "drama" mungkin muncul di banyak film), sehingga genre yang lebih unik mendapat bobot lebih tinggi.

Hasil dari proses ini adalah **matriks TF-IDF**, di mana setiap baris merepresentasikan satu film dan setiap kolom mewakili satu genre. Nilai-nilai dalam matriks ini menunjukkan seberapa kuat keterkaitan film tersebut dengan masing-masing genre.

Langkah pertama adalah mengubah teks pada kolom `genres` menjadi representasi numerik menggunakan **TF-IDF (Term Frequency - Inverse Document Frequency)**. Proses ini bertujuan untuk mengetahui seberapa penting suatu genre (seperti *action*, *drama*, *comedy*, dll.) dalam keseluruhan kumpulan data film:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TF-IDF
tfidf = TfidfVectorizer()

# Fit dan transform kolom genres menjadi matriks TF-IDF
tfidf_matrix = tfidf.fit_transform(movie_merged['genres'])

# Melihat ukuran matrix tfidf
tfidf_matrix.shape
```
> (992, 22)

```
# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()
```

**Hasil Output :**

```
matrix([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.69140498, 0.        ,
         0.        ],
        [0.        , 0.59553039, 0.        , ..., 0.        , 0.        ,
         0.        ],
        ...,
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.52798862, ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ]])
```

> **Insight:**
> TF-IDF berhasil mengubah data genre dari 992 film menjadi matriks berukuran 992 × 22, di mana setiap baris mewakili satu film dan setiap kolom mewakili bobot penting genre tertentu. Ini memungkinkan analisis numerik berbasis teks yang lebih akurat.

```
# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan jenis genre
# Baris diisi dengan judul movie

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names_out(),
    index=movie_merged.title
).sample(22, axis=1).sample(10, axis=0)
```

**Output :**
```
drama	action	documentary	adventure	fiction	fantasy	western	family	romance	foreign	...	mystery	horror	science	movie	tv	animation	crime	comedy	thriller	music
title																					
48 Hrs.	0.302745	0.474558	0.0	0.000000	0.000000	0.000000	0.0	0.0	0.0	0.0	...	0.000000	0.0	0.000000	0.0	0.0	0.0	0.526189	0.443827	0.457475	0.0
Men in Black II	0.000000	0.408583	0.0	0.435793	0.498561	0.000000	0.0	0.0	0.0	0.0	...	0.000000	0.0	0.498561	0.0	0.0	0.0	0.000000	0.382124	0.000000	0.0
Murder She Said	0.291891	0.000000	0.0	0.000000	0.000000	0.000000	0.0	0.0	0.0	0.0	...	0.688702	0.0	0.000000	0.0	0.0	0.0	0.507324	0.427915	0.000000	0.0
The Green Mile	0.351208	0.000000	0.0	0.000000	0.000000	0.709958	0.0	0.0	0.0	0.0	...	0.000000	0.0	0.000000	0.0	0.0	0.0	0.610420	0.000000	0.000000	0.0
Citizen Kane	0.390226	0.000000	0.0	0.000000	0.000000	0.000000	0.0	0.0	0.0	0.0	...	0.920719	0.0	0.000000	0.0	0.0	0.0	0.000000	0.000000	0.000000	0.0
The Green Mile	0.351208	0.000000	0.0	0.000000	0.000000	0.709958	0.0	0.0	0.0	0.0	...	0.000000	0.0	0.000000	0.0	0.0	0.0	0.610420	0.000000	0.000000	0.0
Tough Enough	0.551873	0.000000	0.0	0.000000	0.000000	0.000000	0.0	0.0	0.0	0.0	...	0.000000	0.0	0.000000	0.0	0.0	0.0	0.000000	0.000000	0.833928	0.0
Indiana Jones and the Last Crusade	0.000000	0.683965	0.0	0.729515	0.000000	0.000000	0.0	0.0	0.0	0.0	...	0.000000	0.0	0.000000	0.0	0.0	0.0	0.000000	0.000000	0.000000	0.0
The Perfect Storm	1.000000	0.000000	0.0	0.000000	0.000000	0.000000	0.0	0.0	0.0	0.0	...	0.000000	0.0	0.000000	0.0	0.0	0.0	0.000000	0.000000	0.000000	0.0
Bad Boys II	0.000000	0.439755	0.0	0.469041	0.000000	0.000000	0.0	0.0	0.0	0.0	...	0.000000	0.0	0.000000	0.0	0.0	0.0	0.487600	0.411278	0.423925	0.0
10 rows × 22 columns
```

**Insight :**

Movie *The Perfect Storm* memiliki genre *war*, movie *Citizen Kane* memiliki genre *mystery*, movie *Indiana Jones and the Last Crusade* memiliki genre *adventure*.

TF-IDF berhasil membantu mengidentifikasi genre paling khas di tiap film.

---

## **Model Development: Content-Based Filtering menggunakan Cosine Similarity**

Untuk membangun sistem rekomendasi film berbasis konten, pendekatan yang digunakan adalah **content-based filtering**, yang merekomendasikan film berdasarkan kemiripan kontennya — dalam hal ini adalah **genre**. Langkah-langkah pengembangannya dijelaskan sebagai berikut:

---

### **1. Penggunaan Cosine Similarity untuk Mengukur Kemiripan Antar Film**

Untuk membangun sistem rekomendasi berbasis konten, pendekatan yang digunakan adalah **content-based filtering**, di mana rekomendasi diberikan berdasarkan kemiripan konten antar item, dalam hal ini adalah **genre** film. Genre setiap film direpresentasikan dalam bentuk vektor menggunakan metode **TF-IDF (Term Frequency-Inverse Document Frequency)**. Vektor tersebut kemudian dibandingkan satu sama lain untuk mengukur tingkat kemiripan.

Ukuran kemiripan yang digunakan adalah **cosine similarity**, yang menghitung seberapa kecil sudut antara dua vektor dalam ruang multidimensi. Karakteristik dari cosine similarity:

* Nilainya berkisar dari **0** hingga **1**, di mana nilai **1** menunjukkan dua vektor sangat mirip atau identik, dan **0** berarti tidak ada kemiripan.
* Dalam konteks ini, dua film dianggap mirip jika genre-nya memiliki distribusi kata yang serupa dalam representasi TF-IDF-nya.

Dengan pendekatan ini, sistem dapat mengidentifikasi film-film yang paling relevan berdasarkan genre dari film input, kemudian menyarankan film-film yang memiliki konten paling mirip.

---

### **2. Fungsi `movie_recommendations()` untuk Menghasilkan Rekomendasi**

Fungsi `movie_recommendations()` bertanggung jawab untuk menghasilkan daftar rekomendasi film berdasarkan hasil perhitungan cosine similarity. Berikut adalah penjelasan alur kerja dari fungsi tersebut:

* **Parameter yang digunakan**:

  * `title`: Judul film yang dijadikan referensi awal untuk mencari kemiripan.
  * `similarity`: Matriks cosine similarity antar film, di mana nilai di dalamnya menunjukkan tingkat kemiripan antara setiap pasangan film.
  * `items`: DataFrame yang berisi informasi film (judul dan genre).
  * `top_n`: Jumlah film yang ingin direkomendasikan.

* **Langkah-langkah dalam fungsi**:

  1. Mengecek apakah judul film input tersedia dalam kolom matriks similarity. Jika tidak ditemukan, fungsi akan menghentikan proses dengan menampilkan error.
  2. Menentukan batas maksimum rekomendasi (`top_n`), disesuaikan agar tidak melebihi jumlah film dalam dataset.
  3. Mengambil indeks film-film dengan nilai cosine similarity tertinggi terhadap film input menggunakan metode `argpartition` — pendekatan ini efisien untuk pencarian elemen terbesar.
  4. Menghapus film itu sendiri dari hasil rekomendasi agar tidak direkomendasikan ke pengguna.
  5. Menggabungkan hasil rekomendasi dengan DataFrame `items` untuk menampilkan informasi judul dan genre film.
  6. Mengembalikan `top_n` film yang paling mirip sebagai hasil rekomendasi.

Dengan demikian, fungsi ini menjadi inti dari sistem rekomendasi berbasis konten yang mengandalkan kemiripan genre untuk menyarankan film-film relevan kepada pengguna.

### **a. Cosine Similarity**

Setelah setiap film direpresentasikan sebagai vektor TF-IDF, langkah berikutnya adalah mengukur tingkat kemiripan antar film. Untuk ini digunakan **cosine similarity**, yaitu ukuran kesamaan antara dua vektor berdasarkan sudut di antara mereka.

* Nilai cosine similarity berkisar antara **0** (tidak mirip sama sekali) hingga **1** (sangat mirip).
* Dua film dianggap mirip jika genre-nya memiliki distribusi yang serupa dalam representasi TF-IDF mereka.

Dengan menggunakan cosine similarity, sistem dapat mencari film yang paling dekat (mirip) vektornya dengan film input, dan merekomendasikannya ke pengguna. Setiap film direpresentasikan dalam bentuk vektor yang menggambarkan genre-genre yang dimiliki, kemudian dihitung sudut kosinus antara vektor-vektor tersebut untuk menentukan seberapa mirip kedua film tersebut secara genre.

```
from sklearn.metrics.pairwise import cosine_similarity

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim
```

**Hasil Output :**
```
array([[1.        , 0.20956695, 0.35619879, ..., 0.56350998, 0.        ,
        0.56350998],
       [0.20956695, 1.        , 0.07464749, ..., 0.11809307, 0.        ,
        0.11809307],
       [0.35619879, 0.07464749, 1.        , ..., 0.20072157, 0.27165012,
        0.20072157],
       ...,
       [0.56350998, 0.11809307, 0.20072157, ..., 1.        , 0.22602446,
        1.        ],
       [0.        , 0.        , 0.27165012, ..., 0.22602446, 1.        ,
        0.22602446],
       [0.56350998, 0.11809307, 0.20072157, ..., 1.        , 0.22602446,
        1.        ]])
```

**Insight :**

Matriks tersebut menunjukkan **kemiripan antar film berdasarkan genre** dalam bentuk nilai cosine similarity.

* Nilai 1 berarti film sangat mirip (biasanya terhadap dirinya sendiri).
* Nilai mendekati 0 berarti sangat tidak mirip.
* Contoh: jika baris ke-0 dan kolom ke-3 bernilai 0.56, artinya film 0 dan film 3 punya genre yang cukup mirip.


```python
from sklearn.metrics.pairwise import cosine_similarity

# Menghitung kemiripan antar film
cosine_sim = cosine_similarity(tfidf_matrix)

# Mengubah ke DataFrame untuk memudahkan interpretasi
cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_merged['title'], columns=movie_merged['title'])

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa judul movie
cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_merged['title'], columns=movie_merged['title'])
print('Shape:', cosine_sim_df.shape)

# Melihat similarity matrix pada setiap movie
cosine_sim_df.sample(5, axis=1).sample(20, axis=0)
```

| title                   | Let's Get Harry | Tough Enough | That Man from Rio | Metropolis | Roustabout |
|-------------------------|-----------------|--------------|-------------------|------------|------------|
| Grill Point             | 0.000000        | 0.310986     | 0.445156          | 0.195398   | 0.156908   |
| Basquiat                | 0.000000        | 0.186034     | 0.000000          | 0.116889   | 0.093863   |
| The Man in the Iron Mask | 0.000000        | 0.255213     | 0.000000          | 0.160355   | 0.602113   |
| Jurassic Park           | 0.383551        | 0.000000     | 0.323102          | 0.797856   | 0.000000   |
| The Green Mile          | 0.000000        | 0.193822     | 0.000000          | 0.121782   | 0.097793   |
| The Host                | 0.000000        | 0.151969     | 0.000000          | 0.794141   | 0.076676   |
| Rocky II                | 0.000000        | 0.551873     | 0.000000          | 0.346752   | 0.278447   |
| Beauty and the Beast    | 0.000000        | 0.186436     | 0.000000          | 0.117141   | 0.439850   |
| Rocky V                 | 0.000000        | 0.551873     | 0.000000          | 0.346752   | 0.278447   |
| Rome, Open City         | 0.000000        | 0.186034     | 0.000000          | 0.116889   | 0.093863   |
| The Discovery of Heaven | 0.000000        | 0.244701     | 0.000000          | 0.153750   | 0.123464   |
| That Man from Rio       | 0.842396        | 0.000000     | 1.000000          | 0.000000   | 0.000000   |
| The Tunnel              | 0.000000        | 0.000000     | 0.000000          | 0.937957   | 0.000000   |
| The Machinist           | 0.000000        | 1.000000     | 0.000000          | 0.191363   | 0.153667   |
| Beauty and the Beast    | 0.000000        | 0.186436     | 0.000000          | 0.117141   | 0.439850   |
| The Hours               | 0.000000        | 0.551873     | 0.000000          | 0.346752   | 0.278447   |
| Spy Game                | 0.384837        | 0.452325     | 0.324185          | 0.000000   | 0.000000   |
| Reign Over Me           | 0.000000        | 0.551873     | 0.000000          | 0.346752   | 0.278447   |
| Metropolis              | 0.000000        | 0.191363     | 0.000000          | 1.000000   | 0.096552   |
| Ocean's Eleven          | 0.000000        | 0.318282     | 0.199528          | 0.000000   | 0.578234   |

> **Insight:**

Setiap sel menunjukkan seberapa mirip dua film berdasarkan TF-IDF genre-nya:

- Nilai 1 berarti film yang sama atau sangat mirip.

- Nilai mendekati 0 berarti tidak mirip.

Berdasarkan matriks similarity yang ditampilkan, film *The Machinist* memiliki nilai kemiripan tertinggi dengan film *Tough Enough	* sebesar 1.0, yang menunjukkan kesamaan genre atau fitur lainnya yang sangat kuat antara kedua film tersebut. 

---

### 2. **Movie Recommendation Function**

Fungsi berikut dibuat untuk menghasilkan rekomendasi film berdasarkan judul film yang dimasukkan. Proses utama yang terjadi di fungsi ini adalah:

1. **Validasi Judul Film Input**
   Fungsi mulai dengan memastikan film yang dimasukkan sebagai referensi (`title`) ada di dalam data cosine similarity matrix. Kalau gak ada, fungsi langsung kasih error supaya tahu film itu tidak tersedia untuk rekomendasi.

2. **Penentuan Jumlah Rekomendasi yang Realistis**
   Fungsi membatasi jumlah film rekomendasi yang akan dikembalikan supaya tidak melebihi jumlah film yang ada di data (dikurangi 1 karena film yang direferensikan gak masuk rekomendasi).

3. **Mencari Film yang Paling Mirip dengan Film Referensi**
   Fungsi menggunakan metode `argpartition` pada array nilai similarity untuk menemukan film-film yang memiliki skor kemiripan tertinggi dengan film referensi. Metode ini lebih efisien daripada mengurutkan semua nilai similarity.

4. **Mengambil Judul Film Film yang Paling Mirip**
   Setelah menemukan indeks film-film dengan skor kemiripan tertinggi, fungsi mengambil judul film-film tersebut dari kolom matrix similarity.

5. **Menghapus Film Referensi dari Daftar Rekomendasi**
   Film referensi tidak dimasukkan ke dalam rekomendasi karena sudah pasti sama dengan film yang jadi acuan.

6. **Menggabungkan Data Film dengan Info Genre**
   Fungsi menggabungkan judul film-film yang direkomendasikan dengan data film asli (`items`) supaya hasil rekomendasi gak cuma judul, tapi juga menampilkan genre film.

7. **Mengembalikan Data Frame Rekomendasi**
   Fungsi mengembalikan sebuah DataFrame yang berisi film-film rekomendasi (judul dan genre) sebanyak `top_n` film yang paling mirip.

```python
def movie_recommendations(title, similarity=cosine_sim_df, items=movie_merged[['title', 'genres']], top_n=5):
    """
    Rekomendasi movie berdasarkan kemiripan genre menggunakan cosine similarity.
    
    Parameters:
    - title (str): Judul film yang dijadikan referensi.
    - similarity (DataFrame): Matrix cosine similarity antar film.
    - items (DataFrame): Dataframe berisi informasi judul dan genre film.
    - top_n (int): Jumlah rekomendasi film yang dihasilkan.
    
    Returns:
    - DataFrame: Rekomendasi film beserta genrenya.
    """
    if title not in similarity.columns:
        raise ValueError(f"Title '{title}' tidak ditemukan di similarity matrix.")

    # Batas maksimum rekomendasi yang masuk akal (tidak termasuk dirinya sendiri)
    top_n = min(top_n, similarity.shape[0] - 1)

    # Cari index film paling mirip (dengan argpartition untuk efisiensi)
    index = similarity.loc[:, title].to_numpy().argpartition(range(-1, -top_n-1, -1)).flatten()
    most_similar = similarity.columns[index[-1:-(top_n+2):-1]]

    # Hapus film itu sendiri dari hasil rekomendasi
    most_similar = most_similar.drop(title, errors='ignore')

    return (
        pd.DataFrame(most_similar, columns=['title'])
        .merge(items.drop_duplicates('title'), on='title')
        .head(top_n)
    )
```

## Kesimpulan Fungsi `movie_recommendations`

Fungsi ini bertujuan untuk **memberikan rekomendasi film berdasarkan kemiripan genre film yang sudah ada**. Kemiripan ini dihitung menggunakan **cosine similarity**, yaitu sebuah metode yang mengukur seberapa mirip dua film berdasarkan fitur (dalam kasus ini genre) yang mereka miliki.

---

Contoh penggunaan function movie_recommendation dengan top_n = 10 dengan judul 'Beauty and The Beast' :
```
movie_recommendations('Beauty and the Beast', top_n = 10)
```

**Output Rekomendasi :***
|    | Title                                         	 | Genres                                 		  |
|----|---------------------------------------------------|--------------------------------------------------------|
| 0  | Jurassic Park                                 	 | Adventure \| Science Fiction          		  |
| 1  | The Golem: How He Came Into the World         	 | Horror \| Science Fiction \| Thriller 		  |
| 2  | Armageddon                                    	 | Action \| Thriller \| Science Fiction \| Adventure     |
| 3  | The Hours                                     	 | Drama                                 		  |
| 4  | The Getaway                                  	 | Drama \| Action \| Thriller           		  |
| 5  | Grill Point                                  	 | Comedy \| Drama                        		  |
| 6  | Love Is the Devil: Study for a Portrait of Fra... | TV Movie \| Drama                     	 	  |
| 7  | The Last Laugh                                	 | Drama                                 	  	  |
| 8  | The Tunnel                                   	 | Science Fiction                      		  |
| 9  | Tough Enough                                  	 | Drama \| Thriller                      		  |

> **Insight:**

> Tabel ini berisi daftar film beserta genre-nya yang bisa digunakan sebagai dasar untuk sistem rekomendasi film berbasis genre. Dengan menggunakan *top\_n = 10*, artinya sistem akan merekomendasikan 10 film paling mirip berdasarkan kesamaan genre dengan film referensi. Genre campuran seperti “Adventure | Science Fiction” atau “Drama | Action | Thriller” menunjukkan bahwa film-film ini bisa memiliki overlap genre yang kompleks, sehingga cosine similarity dapat menangkap tingkat kemiripan yang lebih detail daripada hanya mencocokkan genre tunggal. Ini memungkinkan rekomendasi yang lebih relevan dan beragam bagi pengguna.

---

## **Kelebihan Content-Based Filtering**

1. **Personalisasi tinggi**

   Rekomendasi disesuaikan dengan preferensi unik setiap pengguna berdasarkan item yang pernah mereka sukai.
   *Contoh:* Jika seorang pengguna menyukai film *Inception* karena bergenre Sci-Fi dan Thriller, maka sistem akan merekomendasikan film seperti *Interstellar* atau *Tenet* yang memiliki genre serupa.

2. **Tidak butuh data dari pengguna lain**

   Sistem hanya mengandalkan informasi dari item (misalnya genre film), sehingga cocok digunakan saat jumlah pengguna masih sedikit (*cold start for user*).
   *Contoh:* Sistem tetap bisa bekerja meski hanya ada satu pengguna karena cukup melihat genre film yang disukai.

3. **Tahan terhadap serangan manipulasi user**

   Karena tidak bergantung pada rating dari banyak pengguna, sistem ini relatif aman dari spam atau rating palsu.
   *Contoh:* Tidak terpengaruh jika ada banyak akun palsu yang memberi rating tinggi ke film tertentu.

---

## **Kekurangan Content-Based Filtering**

1. **Rekomendasi cenderung sempit (kurang variatif)**

   Hanya merekomendasikan item yang sangat mirip, sehingga pengguna bisa merasa terjebak dalam "filter bubble".
   *Contoh:* Jika pengguna suka film horor, sistem hanya akan merekomendasikan horor terus-menerus, meskipun mungkin ia juga akan suka thriller atau mystery.

2. **Sulit menangkap selera kompleks**

   Jika preferensi pengguna mencakup banyak genre atau aspek lain (seperti sutradara atau alur cerita), pendekatan ini bisa kurang fleksibel.
   *Contoh:* Seorang pengguna menyukai *The Dark Knight* bukan hanya karena genre-nya, tapi juga karena sutradaranya (Christopher Nolan), namun sistem hanya melihat genre "Action" dan "Crime".

3. **Ketergantungan pada representasi fitur konten**

   Jika fitur kontennya tidak lengkap atau tidak akurat, maka kualitas rekomendasinya akan menurun.
   *Contoh:* Jika informasi genre sebuah film tidak ditulis lengkap, maka sistem tidak bisa memberikan rekomendasi yang tepat.

---

## **Evaluasi**

Untuk menilai kinerja dari model **content-based filtering** dalam sistem rekomendasi film ini, digunakan tiga metrik utama yang berbasis relevansi, yaitu **Precision\@k**, **Recall\@k**, dan **F1-Score\@k**. Ketiga metrik ini umum digunakan dalam evaluasi sistem rekomendasi karena dapat mengukur seberapa tepat dan lengkap rekomendasi yang diberikan.

1. **Precision\@k**
   Precision\@k mengukur seberapa banyak item yang benar-benar relevan di antara *k item* teratas yang direkomendasikan oleh sistem. Metrik ini menggambarkan **tingkat akurasi** rekomendasi.

$$
\text{Precision@k} = \frac{\text{Jumlah item relevan dalam rekomendasi}}{k}
$$

2. **Recall\@k**
   Recall\@k menunjukkan sejauh mana sistem berhasil menemukan item yang relevan dari seluruh item relevan yang ada. Artinya, metrik ini menilai **cakupan** dari sistem rekomendasi.

$$
\text{Recall@k} = \frac{\text{Jumlah item relevan dalam rekomendasi}}{\text{Jumlah total item relevan}}
$$

3. **F1-Score\@k**
   F1-Score\@k merupakan kombinasi dari Precision dan Recall dalam bentuk rata-rata harmonik. Metrik ini digunakan untuk menilai **keseimbangan antara ketepatan dan kelengkapan** dalam hasil rekomendasi.

$$
\text{F1-Score@k} = \frac{2 \times \text{Precision@k} \times \text{Recall@k}}{\text{Precision@k} + \text{Recall@k}}
$$

---

```
# data ground truth (film relevan yang disukai user)
ground_truth = {
    "Live and Let Die": {"The Getaway", "Dr. Jekyll and Mr. Hyde", "Big Fish"},
}

def evaluate_recommendation(ground_truth, k=5):
    precision_list = []
    recall_list = []
    f1_list = []

    for title, relevant_set in ground_truth.items():
        # Ambil rekomendasi film dari fungsi
        recommended_df = movie_recommendations(title, top_n=k)
        recommended = set(recommended_df['title'])
        
        true_positives = recommended & relevant_set
        
        precision = len(true_positives) / k if k > 0 else 0
        recall = len(true_positives) / len(relevant_set) if len(relevant_set) > 0 else 0
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    print(f"Average Precision@{k}: {sum(precision_list)/len(precision_list):.2f}")
    print(f"Average Recall@{k}: {sum(recall_list)/len(recall_list):.2f}")
    print(f"Average F1-Score@{k}: {sum(f1_list)/len(f1_list):.2f}")
    print("Relevant Movies :", list(true_positives))

evaluate_recommendation(ground_truth, k=5)
```

**Hasil Output :**
```
Average Precision@5: 0.60
Average Recall@5: 1.00
Average F1-Score@5: 0.75
Relevant Movies : ['The Getaway', 'Dr. Jekyll and Mr. Hyde', 'Big Fish']
```

**Insight :**

- **Precision@5 sebesar 0.60** menunjukkan bahwa dari lima film yang direkomendasikan oleh sistem, rata-rata tiga film di antaranya merupakan film yang relevan atau sesuai dengan preferensi pengguna. Dengan kata lain, 60% rekomendasi yang diberikan tepat sasaran.

- **Recall@5 sebesar 1.00** mengindikasikan bahwa semua film yang dianggap relevan (berdasarkan data ground truth) berhasil direkomendasikan oleh sistem dalam lima rekomendasi teratas. Ini menunjukkan bahwa sistem memiliki kemampuan yang sangat baik dalam menemukan semua item relevan yang tersedia.

- **F1-Score@5 sebesar 0.75** merupakan nilai rata-rata harmonik dari precision dan recall, yang mencerminkan keseimbangan yang baik antara ketepatan dan kelengkapan rekomendasi yang diberikan oleh sistem.

---

# **Problem Answers**

---

## **1. Bagaimana cara merepresentasikan informasi genre film secara numerik agar bisa digunakan dalam perhitungan kemiripan antar film?**

```python
# Cek apakah TF-IDF berhasil transform genre jadi matriks numerik
try:
    tfidf_matrix = tfidf.fit_transform(movie_merged['genres'])
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    assert tfidf_matrix.shape[0] == movie_merged.shape[0], "Jumlah baris TF-IDF harus sama dengan jumlah film"
    print("Representasi genre ke numerik berhasil.")
except Exception as e:
    print("Error pada representasi genre:", e)
````

**Hasil Output:**

```
TF-IDF matrix shape: (992, 22)
Representasi genre ke numerik berhasil.
```

**Insight :**

TF-IDF berhasil mengubah genre film menjadi vektor numerik berdimensi **(992, 22)**, artinya **992 film** direpresentasikan oleh **22 genre unik**. Representasi ini memungkinkan perhitungan kemiripan antar film berdasarkan bobot genre, dan jadi dasar utama untuk sistem rekomendasi content-based filtering.

---

## **2. Bagaimana cara mengukur tingkat kemiripan antar film hanya berdasarkan informasi kontennya, khususnya genre?**

```python
# Tentukan dua judul film yang ingin dibandingkan
movie_1 = 'Men in Black II'
movie_2 = 'Jurassic Park'

# Ambil nilai similarity antara dua film tersebut
similarity_score = cosine_sim_df.loc[movie_1, movie_2]

# Konversi ke float kalau hasilnya satu sel DataFrame
if isinstance(similarity_score, pd.DataFrame):
    similarity_score = similarity_score.values[0][0]
elif isinstance(similarity_score, pd.Series):
    similarity_score = similarity_score.values[0]

print(f"Similarity antara '{movie_1}' dan '{movie_2}': {similarity_score:.4f}")
```

**Hasil Output:**

```
Similarity antara 'Men in Black II' dan 'Jurassic Park': 0.8289
```

**Insight :**

Dengan menghitung **cosine similarity** dari representasi TF-IDF genre, kita bisa mengukur seberapa mirip dua film berdasarkan kontennya. Nilai similarity **0.8289** menunjukkan bahwa *Men in Black II* dan *Jurassic Park* memiliki genre yang sangat mirip, seperti action, adventure, atau sci-fi. Ini jadi dasar sistem rekomendasi berbasis konten.

---

## **3. Bagaimana cara mengembangkan sistem rekomendasi film yang mampu memberikan saran film sejenis hanya dari satu input judul film?**

```python
# Fungsi inputan dan rekomendasi film
def test_movie_recommendation():
    title_input = input("Masukkan judul film yang ingin direkomendasikan film miripnya: ")
    try:
        recommendations = movie_recommendations(title_input, top_n=5)
        print(f"\nRekomendasi film mirip dengan '{title_input}':\n")
        print(recommendations[['title', 'genres']])
    except KeyError:
        print(f"Judul film '{title_input}' tidak ditemukan dalam data.")

# Jalankan fungsi uji rekomendasi dengan inputan
test_movie_recommendation()
```

**Contoh Input:**

```
Masukkan judul film yang ingin direkomendasikan film miripnya: Armageddon
```

**Hasil Output:**

```
Rekomendasi film mirip dengan 'Armageddon':

                     title                            genres
0              Grill Point                    Comedy | Drama
1             Tough Enough                  Drama | Thriller
2           The Last Laugh                             Drama
3  Dr. Jekyll and Mr. Hyde  Drama | Horror | Science Fiction
4                 Big Fish       Adventure | Fantasy | Drama
```

**Insight :**

Model berhasil memberikan rekomendasi film berdasarkan kemiripan genre.

Film seperti Grill Point, Tough Enough, dan Big Fish direkomendasikan karena memiliki genre yang serupa, seperti Drama, Adventure, atau Thriller, sesuai dengan pendekatan content-based filtering. Ini menunjukkan bahwa sistem dapat menemukan film sejenis hanya dari input satu judul film.

---

## **Referensi**

* Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In *Recommender Systems Handbook* (pp. 1–35). Springer. [https://doi.org/10.1007/978-0-387-85820-3\_1](https://doi.org/10.1007/978-0-387-85820-3_1)

* Aggarwal, C. C. (2016). Content-based recommender systems. In *Recommender Systems: The Textbook* (pp. 139–166). Springer. [https://doi.org/10.1007/978-3-319-29659-3\_5](https://doi.org/10.1007/978-3-319-29659-3_5)

* Banik, R. (2017). *The Movies Dataset*. Kaggle. [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)



