# Laporan Proyek Machine Learning - Sekar Ayu Widhastri

## Project Overview

Industri kecantikan dan perawatan kulit (skincare) merupakan salah satu pasar dengan pertumbuhan tercepat dan paling dinamis di dunia. Menurut riset dari Statista, pasar kecantikan global diproyeksikan akan mencapai nilai sekitar 580 miliar Dolar AS pada tahun 2027 [1]. Pertumbuhan ini didorong oleh inovasi produk yang tiada henti dan meningkatnya kesadaran konsumen akan perawatan diri. Namun, ledakan jumlah produk ini menciptakan sebuah tantangan besar bagi konsumen, yaitu paradoks pilihan (paradox of choice). Konsumen sering kali merasa bingung dan kewalahan saat dihadapkan pada ribuan pilihan produk yang tampak serupa, membuat proses pengambilan keputusan menjadi sulit dan berisiko. Kesalahan dalam memilih produk tidak hanya menyebabkan kerugian finansial, tetapi juga berpotensi menimbulkan masalah pada kulit.

Untuk mengatasi masalah ini, personalisasi menjadi kunci. Sistem rekomendasi hadir sebagai solusi teknologi yang mampu menyaring ribuan pilihan dan menyajikan produk yang paling relevan untuk setiap individu. Dengan menganalisis data interaksi pengguna dan atribut produk, sistem ini dapat memberikan saran yang dipersonalisasi, meningkatkan pengalaman berbelanja, membangun loyalitas pelanggan, dan pada akhirnya meningkatkan penjualan bagi platform e-commerce. Proyek ini bertujuan untuk membangun sistem rekomendasi produk skincare yang cerdas dengan menggabungkan beberapa pendekatan untuk memberikan saran yang akurat dan relevan dalam berbagai skenario pengguna

## Business Understanding
### Problem Statements

Berdasarkan latar belakang tersebut, masalah utama yang ingin diselesaikan dalam proyek ini dapat dirumuskan sebagai berikut:

1. Bagaimana cara membantu pengguna menemukan produk lain yang memiliki karakteristik (kandungan, fungsi, kategori) serupa dengan produk yang sedang mereka minati?
2. Bagaimana cara memberikan rekomendasi produk yang benar-benar personal, berdasarkan riwayat penilaian dan selera unik dari setiap pengguna?
3. Bagaimana cara memberikan panduan kepada pengguna baru (yang belum memiliki riwayat aktivitas) atau pengguna yang hanya ingin melihat produk-produk terbaik secara umum?

### Goals

Untuk menjawab rumusan masalah di atas, tujuan dari proyek ini adalah:

1. Mengembangkan model rekomendasi yang mampu menyarankan produk berdasarkan kemiripan konten (Content-Based Filtering).
2. Mengembangkan model rekomendasi yang dapat mempelajari preferensi laten pengguna dari data historis untuk memberikan rekomendasi personal (Collaborative Filtering).
3. Mengembangkan sistem yang dapat menampilkan produk-produk paling populer atau dengan rating tertinggi sebagai rekomendasi umum (Popularity-Based Filtering).

### Solution statements

Untuk mencapai tujuan-tujuan tersebut, proyek ini akan mengimplementasikan tiga pendekatan solusi:

1. Content-Based Filtering: Menggunakan algoritma TF-IDF Vectorizer untuk mengubah atribut teks produk menjadi vektor numerik, lalu menghitung kemiripan antar produk menggunakan Cosine Similarity.
2. Popularity-Based Filtering: Membuat sistem peringkat sederhana dengan menghitung skor popularitas untuk setiap produk berdasarkan rating rata-rata dan jumlah "loves" yang diterima.
3. Collaborative Filtering: Mengimplementasikan model faktorisasi matriks menggunakan arsitektur jaringan saraf tiruan (neural network) dengan TensorFlow/Keras. Pendekatan ini akan mempelajari embedding (representasi laten) untuk setiap pengguna dan produk untuk memprediksi rating.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah "Sephora Products and Skincare Reviews" yang bersumber dari Kaggle. Dataset ini terdiri dari dua bagian utama: informasi detail produk dan ulasan dari para pengguna. Setelah proses pemuatan dan penggabungan, dataset utama berisi 322,987 baris data ulasan yang telah diperkaya dengan informasi produk terkait.

Variabel-variabel utama yang digunakan setelah proses pembersihan adalah sebagai berikut:

- author_id: ID unik untuk setiap pengguna/penulis ulasan.
- user_rating: Rating yang diberikan oleh pengguna pada produk (skala 1-5).
- product_id: ID unik untuk setiap produk.
- product_name: Nama lengkap produk.
- brand_name: Nama brand dari produk.
- loves_count: Jumlah "loves" atau "likes" yang diterima produk, sebagai indikator popularitas.
- avg_product_rating: Rating rata-rata yang dimiliki produk secara keseluruhan.
- ingredients: Daftar bahan atau kandungan dari produk.
- category: Gabungan dari kategori primer, sekunder, dan tersier produk.

### Exploratory Data Analysis
Untuk memahami data lebih dalam, dilakukan beberapa visualisasi:

1. Distribusi Rating Pengguna
   ![image](https://github.com/user-attachments/assets/e2f1e781-a131-41b2-9f8e-047e00270b2b)

   _Insight_: Terdapat bias positif yang kuat, di mana sebagian besar pengguna memberikan rating 5. Hal ini menandakan pengguna cenderung mengulas produk yang mereka sukai, atau kualitas produk di platform memang secara umum tinggi.

2. Top 10 Brand dengan Ulasan Terbanyak
   ![image](https://github.com/user-attachments/assets/325ee76d-8ce6-4776-b65b-039e11fc55d8)

   _Insight_: Brand seperti Peter Thomas Roth, Murad, dan CLINIQUE sangat dominan dalam dataset. Hal ini berarti data interaksi untuk produk dari brand-brand ini sangat kaya dan dapat diandalkan untuk pemodelan.

3. Distribusi Kategori Produk Utama
   ![image](https://github.com/user-attachments/assets/413e5dc8-ba4a-45f8-8963-61fb1ca7864c)

   _Insight_: Terlihat bahwa Skincare adalah kategori yang paling dominan. Hal ini mengkonfirmasi bahwa dataset ini sangat relevan untuk membuat sistem rekomendasi produk perawatan kulit.

4. Distribusi Tipe Kulit Pengguna
   ![image](https://github.com/user-attachments/assets/1c0dcba5-d63d-489c-b825-49e2f97929bd)

   _Insight_: Tipe kulit kombinasi, normal, dan berminyak adalah yang paling umum di antara pengguna, dengan tipe kulit combination (kombinasi) adalah yang paling banyak, diikuti oleh dry (kering) dan normal. Informasi ini sangat berharga untuk segmentasi atau personalisasi lebih lanjut di masa depan.
      
5. Distribusi Rata-rata Produk dan "Loves" Count
   ![image](https://github.com/user-attachments/assets/b8fb578e-e843-4ea3-9f78-120487cf5106)

   _Insight_: Terlihat bahwa ada tren positif yang terlihat; produk dengan rating rata-rata lebih tinggi cenderung memiliki jumlah "Loves" yang lebih tinggi pula. Penggunaan skala logaritmik pada sumbu Y membantu melihat sebaran data yang sangat bervariasi. Hal ini mengkonfirmasi bahwa loves_count adalah metrik yang baik untuk mengukur popularitas.
   
## Data Preparation
Proses persiapan data dilakukan secara sistematis untuk memastikan data bersih, terstruktur, dan siap untuk ketiga model. Tahapannya adalah sebagai berikut:
1. Penggabungan Data
   Menyatukan beberapa file ulasan (`reviews_250-500.csv` dan `reviews_500-750.csv`) menjadi satu, lalu menggabungkannya dengan file informasi produk (`product_info.csv`) menggunakan `product_id` sebagai kunci.
2. Pembersihan dan Penyesuaian Kolom
   - Memilih hanya kolom-kolom yang relevan untuk pemodelan.
   - Mengganti nama kolom yang ambigu (misalnya `rating_x` menjadi `user_rating` dan `rating_y` menjadi `avg_product_rating`) untuk kejelasan.
   - Menangani nilai yang hilang (missing values), terutama pada kolom ingredients dengan mengisinya dengan string kosong.
3. Pembuatan `unique_products_df`
   Membuat sebuah DataFrame baru yang hanya berisi satu baris untuk setiap produk unik. Hal ini dilakukan untuk efisiensi saat membangun model Content-Based dan Popularity-Based.
4. Pembuatan Fitur `soup`
   Untuk model Content-Based, dibuat sebuah kolom baru bernama soup yang merupakan gabungan dari semua informasi teks produk (`product_name`, `brand_name`, `category`, `ingredients`). Hal ini dilakukan agar TF-IDF dapat menangkap esensi konten dari setiap produk secara holistik.
   ```python
   def create_soup(x):
       return x['product_name'] + ' ' + x['brand_name'] + ' ' + x['category'] + ' ' + x['ingredients']
   unique_products_df['soup'] = unique_products_df.apply(create_soup, axis=1)
   ```
5. Encoding dan Normalisasi untuk Deep Learning
   - Memetakan setiap `author_id` dan `product_id` yang berupa string ke integer unik (`user_index`, `product_index`). Proses ini adalah syarat agar data dapat diproses oleh layer Embedding di TensorFlow.
   - Menormalisasi kolom user_rating (skala 1-5) ke rentang [0, 1]. Hal ini dilakukan agar cocok dengan fungsi aktivasi sigmoid pada output model RecommenderNet.
   ```python
   y = ((df_clean['user_rating'] - min_rating) / (max_rating - min_rating)).values
   ```

## Modeling
Tiga model dikembangkan untuk menjawab permasalahan yang berbeda.

### Model 1: Content-Based Filtering
Model ini merekomendasikan produk berdasarkan kemiripan atribut. Prosesnya adalah mengubah kolom soup menjadi representasi numerik menggunakan TfidfVectorizer, kemudian menghitung kemiripan antar semua produk menggunakan cosine_similarity.
- Kelebihan: Dapat merekomendasikan produk baru yang belum memiliki rating (mengatasi item cold-start problem) dan rekomendasinya mudah dijelaskan.
- Kekurangan: Terbatas pada fitur yang ada dan kurang mampu memberikan rekomendasi "kejutan" (serendipity).

### Model 2: Popularity-Based Filtering
Model ini adalah yang paling sederhana, merekomendasikan produk yang paling populer. Skor popularitas dihitung menggunakan formula weighted rating yang menyeimbangkan rating rata-rata (avg_product_rating) dengan jumlah "loves" (loves_count).
- Kelebihan: Sangat efektif untuk pengguna baru dan mudah diimplementasikan.
- Kekurangan: Tidak dipersonalisasi sama sekali; semua pengguna menerima rekomendasi yang sama.

### Model 3: Collaborative Filtering (RecommenderNet)
Model ini adalah model paling canggih yang dibangun menggunakan arsitektur jaringan saraf tiruan dengan TensorFlow. Model ini mempelajari representasi laten (embedding) untuk setiap pengguna dan produk.

```python
    class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_products, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        # Inisialisasi layer Embedding untuk user dan produk
        self.user_embedding = layers.Embedding(num_users, embedding_size, ...)
        self.user_bias = layers.Embedding(num_users, 1)
        self.product_embedding = layers.Embedding(num_products, embedding_size, ...)
        self.product_bias = layers.Embedding(num_products, 1)
    def call(self, inputs):
        # Ambil vektor user dan produk
        user_vector = self.user_embedding(inputs[:, 0])
        product_vector = self.product_embedding(inputs[:, 1])
        # Hitung dot product dan tambahkan bias
        dot_user_product = tf.tensordot(user_vector, product_vector, 2)
        x = dot_user_product + self.user_bias(inputs[:, 0]) + self.product_bias(inputs[:, 1])
        return tf.nn.sigmoid(x)
```

- Kelebihan: Mampu menangkap pola preferensi yang kompleks dan memberikan rekomendasi yang sangat personal.
- Kekurangan: Memerlukan data interaksi yang besar dan mengalami kesulitan dengan pengguna atau item baru (user/item cold-start problem).

***Contoh Top-N Recommendation**
```python
Berikut adalah contoh output dari sistem gabungan yang menampilkan rekomendasi dari ketiga model:

======================================================================
      Menampilkan Rekomendasi Gabungan untuk Pengguna: 1930716686
======================================================================

KARENA ANDA MELIHAT: 'Benefiance WrinkleResist24 Pure Retinol Express Smoothing Eye Mask'
----------------------------------------------------------------------
Produk Serupa yang Mungkin Anda Suka:
 1. Capture Totale Firming & Wrinkle-Correcting Cream (Brand: Dior)
 2. Vinosource-Hydra SOS Intense Hydration Moisturizer (Brand: Caudalie)
 3. Capture Totale Super Potent Rich Cream (Brand: Dior)

REKOMENDASI KHUSUS UNTUK ANDA
----------------------------------------------------------------------
Berdasarkan Selera Pengguna Lain yang Mirip dengan Anda:
 1. 7 Day Face Scrub Cream Rinse-Off Formula (Brand: CLINIQUE)
 2. Rinse-Off Foaming Cleanser (Brand: CLINIQUE)
 3. Clarifying Lotion 3 (Brand: CLINIQUE)

PRODUK PALING POPULER SAAT INI
----------------------------------------------------------------------
Top 3 Produk Terpopuler di Platform Kami:
 1. Cica Sleeping Mask (Loves: 80111)
 2. Strawberry Smooth BHA + AHA Salicylic Acid Serum (Loves: 118258)
 3. Pore Minimizing Instant Detox Mask (Loves: 96395)
```

## Evaluation
Metrik evaluasi utama yang digunakan adalah Root Mean Squared Error (RMSE), yang diterapkan pada model Collaborative Filtering (RecommenderNet) untuk mengukur seberapa akurat prediksi ratingnya.
**Formula RMSE:**

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^2}
$$

Di mana:
- $RMSE_{orig}$: Nilai RMSE dalam skala rating asli (1-5).
- $RMSE_{norm}$: Nilai RMSE hasil normalisasi dari model (nilai 0.235 Anda).
- $R_{max}$: Rating maksimum (nilai 5).
- $R_{min}$: Rating minimum (nilai 1).

Metrik ini bekerja dengan menghitung selisih antara rating prediksi dan rating asli, mengkuadratkannya (agar semua nilai menjadi positif), merata-ratakannya, lalu mengambil akar kuadrat untuk mengembalikannya ke satuan asli. Intinya, RMSE memberitahu kita "rata-rata seberapa besar kesalahan prediksi rating model kita dalam satuan rating". Nilai yang lebih rendah menunjukkan model yang lebih baik.

**Hasil Proyek:**

![image](https://github.com/user-attachments/assets/e33cc8c0-9877-4ecf-96d7-e0a66316ae7b)

Dari hasil pelatihan model RecommenderNet, diperoleh grafik di atas. Terlihat bahwa nilai error pada data validasi (val_root_mean_squared_error) mencapai titik terendahnya di sekitar epoch ke-10 dengan nilai RMSE sekitar 0.235.
Penting untuk diingat bahwa nilai RMSE ini dihitung pada rating yang telah dinormalisasi ke rentang [0, 1]. Untuk menginterpretasikannya dalam skala rating asli (1-5 bintang), kita perlu mengembalikannya:

$RMSE_{orig} = RMSE_{norm} \times (R_{max} - R_{min})$

$RMSE_{\text{asli}} = 0.235 \times (5 - 1) = 0.94$

Hasil ini menunjukkan bahwa, secara rata-rata, prediksi rating yang diberikan oleh model RecommenderNet memiliki kesalahan (deviasi) sekitar 0.94 bintang dari rating yang sebenarnya diberikan oleh pengguna. Ini adalah hasil yang cukup baik dan menunjukkan bahwa model Collaborative Filtering mampu memberikan prediksi yang akurat secara kuantitatif. Penggunaan callback EarlyStopping dan ModelCheckpoint memastikan bahwa model yang digunakan adalah model dari epoch dengan performa terbaik ini, bukan model dari epoch terakhir yang sudah mulai overfitting.
