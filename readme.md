# **Project Overview**

Dengan terus bertambahnya jumlah film yang tersedia di berbagai platform digital, pengguna sering kesulitan menemukan film yang sesuai dengan selera mereka. Salah satu pendekatan populer untuk mengatasi permasalahan ini adalah sistem rekomendasi berbasis konten (_content-based filtering_), di mana rekomendasi diberikan berdasarkan kesamaan karakteristik konten — dalam hal ini, genre film.

Proyek ini bertujuan membangun **sistem rekomendasi film berdasarkan genre** menggunakan pendekatan content-based filtering. Misalnya, jika seorang pengguna menyukai film _King Kong_, maka sistem akan merekomendasikan film lain yang memiliki genre serupa seperti _Action_, _Adventure_, atau _Horror_.

Sistem ini relevan karena membantu pengguna mengeksplorasi film yang mungkin belum pernah mereka tonton namun memiliki karakteristik yang mereka sukai. Selain itu, pendekatan ini tidak memerlukan data interaksi pengguna lain, sehingga cocok untuk kondisi _cold start_ atau sistem baru.

---

# **Business Understanding**

## **Problem Statements**

1. Bagaimana cara merepresentasikan informasi genre film secara numerik agar bisa digunakan dalam perhitungan kemiripan antar film?
2. Bagaimana cara mengukur tingkat kemiripan antar film hanya berdasarkan informasi kontennya, khususnya genre?
3. Bagaimana cara mengembangkan sistem rekomendasi film yang mampu memberikan saran film sejenis hanya dari satu input judul film?

## **Goals**

1. Menghasilkan sistem rekomendasi film yang mampu menyarankan film lain dengan genre yang mirip dari input satu judul film.
2. Memudahkan pengguna menemukan film-film baru sesuai dengan preferensi genre mereka tanpa perlu memberikan penilaian eksplisit.
3. Membuat sistem yang bersifat general dan bisa digunakan tanpa ketergantungan pada data pengguna (_user rating history_).

## **Solution Statements**

1. Menggunakan pendekatan content-based filtering dengan teknik TF-IDF vectorization untuk merepresentasikan genre film.
2. Mengukur kemiripan antar film menggunakan cosine similarity untuk menentukan film-film yang memiliki kemiripan konten.
3. Mengimplementasikan fungsi rekomendasi yang menerima judul film sebagai input dan menghasilkan daftar rekomendasi berdasarkan tingkat kemiripan genre.

---

# **Data Understanding**

**Link Dataset:**  
https://www.kaggle.com/api/v1/datasets/download/rounakbanik/the-movies-dataset

## **Dataset: Ratings**

```python
data_ratings = pd.read_csv("/content/ratings.csv")
data_ratings.info()
```
