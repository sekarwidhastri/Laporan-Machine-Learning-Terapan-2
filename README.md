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
Dataset yang digunakan dalam proyek ini adalah "Sephora Products and Skincare Reviews", yang bersumber dari _platform_ Kaggle. Dataset ini terdiri dari tiga bagian utama: satu set _file_ berisi informasi detail produk (seperti brand, harga, kandungan, dan kategori) dan dua set _file_ berisi ulasan dari pengguna (termasuk ID pengguna, ID produk, dan rating yang diberikan). Pada dataset produk, terdapat 8.494 produk unik. Pada dataset ulasan pertama, terdapat 206.725 ulasan. Kemudian untuk dataset ulasan kedua, terdapat 116.262 ulasan. Masing-masing ketiga dataset awal ini memiliki _missing values_ yang cukup signifikan. Setelah proses pemuatan dan penggabungan awal, dataset mentah (`df_merged`) berisi 322.987 baris data ulasan. Kondisi data mentah ini menunjukkan adanya beberapa tantangan, seperti kolom yang terduplikasi akibat proses _merge_ (ditandai dengan akhiran _x dan _y), serta adanya nilai yang hilang (_missing values_) pada beberapa kolom deskriptif seperti `ingredients` dan `skin_type`, yang semuanya ditangani pada tahap Data Preparation.

Variabel-variabel utama yang terdapat pada dataset produk adalah sebagai berikut:

- product_id: ID unik produk.
- product_name: Nama produk.
- brand_id: ID unik merek.
- brand_name: Nama merek dari produk.
- loves_count: Jumlah pengguna yang menyukai produk (fitur "Love").
- rating: Nilai rata-rata penilaian dari pengguna.
- reviews: Jumlah ulasan yang diterima produk.
- size: Ukuran produk (misalnya: 30ml, 1 oz).
- variation_type: Jenis variasi produk (misalnya: warna, ukuran).
- variation_value: Nilai dari variasi produk (misalnya: “Red”, “Medium”).
- variation_desc: Deskripsi tambahan dari variasi produk.
- ingredients: Daftar bahan yang terkandung dalam produk.
- price_usd: Harga normal produk dalam USD.
- value_price_usd: Harga nilai (value price), jika tersedia.
- sale_price_usd: Harga diskon (jika produk sedang dijual).
- limited_edition: Indikator apakah produk edisi terbatas (1: ya, 0: tidak).
- new: Indikator apakah produk merupakan produk baru (1: ya, 0: tidak).
- online_only: Indikator apakah produk hanya tersedia secara online (1: ya, 0: tidak).
- out_of_stock: Indikator apakah produk sedang tidak tersedia (1: ya, 0: tidak).
- sephora_exclusive: Indikator apakah produk eksklusif di Sephora (1: ya, 0: tidak).
- highlights: Fitur/keunggulan utama produk (biasanya berupa teks ringkasan).
- primary_category: Kategori utama produk (misalnya: Skincare, Makeup).
- secondary_category: Subkategori produk (misalnya: Foundation, Serum).
- tertiary_category: Sub-subkategori atau jenis produk yang lebih spesifik.
- child_count: Jumlah variasi anak dari produk (misalnya warna atau ukuran).
- child_max_price: Harga tertinggi di antara variasi anak produk.
- child_min_price: Harga terendah di antara variasi anak produk.

Variabel-variabel utama yang terdapat pada dataset ulasan adalah sebagai berikut:

- Unnamed: 0: Indeks baris dari DataFrame asli (hasil dari ekspor CSV).
- author_id: ID unik pengguna yang menulis ulasan.
- rating: Nilai rating yang diberikan terhadap produk (dalam skala 1–5).
- is_recommended: Indikator apakah produk direkomendasikan oleh pengguna (1: ya, 0: tidak).
- helpfulness: Skor bantuan dari ulasan (berapa kali ulasan dianggap membantu).
- total_feedback_count: Jumlah total umpan balik (positif + negatif) yang diterima ulasan.
- total_neg_feedback_count: Jumlah umpan balik negatif yang diterima ulasan.
- total_pos_feedback_count: Jumlah umpan balik positif yang diterima ulasan.
- submission_time: Waktu/tanggal ketika ulasan dikirimkan.
- review_text: Isi atau konten teks dari ulasan produk.
- review_title: Judul dari ulasan produk.
- skin_tone: Nada kulit pengguna yang memberikan ulasan.
- eye_color: Warna mata pengguna.
- skin_type: Jenis kulit pengguna (misal: kering, berminyak, kombinasi).
- hair_color: Warna rambut pengguna.
- product_id: ID unik dari produk yang diulas.
- product_name: Nama produk yang diulas.
- brand_name: Nama merek dari produk yang diulas.
- price_usd: Harga produk dalam dolar AS.

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

Tautan Dataset: [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

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

6. Visualisasi Boxplot Fitur Numerik
   ![image](https://github.com/user-attachments/assets/8e48121d-49e0-4372-9691-787a1c4ac6d9)

   _Insight_: Distribusi fitur numerik menunjukkan pola yang sangat miring (skewed) dan didominasi oleh bias positif. Rating dari pengguna (rating_x) sangat tinggi, dengan mayoritas ulasan memberikan nilai 4 atau 5, sedangkan rating rata-rata produk (rating_y) lebih terkonsentrasi di sekitar 4.3, menandakan kualitas produk yang cenderung seragam. Fitur loves_count dan total_feedback_count menunjukkan pola distribusi "long-tail", di mana hanya sedikit produk atau ulasan yang sangat populer, sementara mayoritas lainnya memiliki keterlibatan rendah. Harga produk (price_usd_y) sebagian besar berada di kisaran terjangkau, namun terdapat outlier produk mewah dengan harga sangat tinggi.
   
## Data Preparation
Proses persiapan data dilakukan secara sistematis untuk memastikan data bersih, terstruktur, dan siap untuk ketiga model. Tahapannya adalah sebagai berikut:
1. Penghapusan Kolom Tidak Relevan
   - Proses: Kolom Unnamed: 0 yang muncul setelah proses pemuatan data langsung dihapus dari DataFrame `df_merged`.
   - Alasan: Kolom ini merupakan sisa indeks dari file CSV dan tidak mengandung informasi yang relevan atau berguna untuk analisis maupun pemodelan. Menghapusnya membuat dataset lebih bersih.
     
2. Seleksi dan Penamaan Ulang Fitur
   - Proses: Hanya kolom-kolom yang esensial untuk ketiga model yang dipilih dari `df_merged`. Kolom-kolom ini kemudian diberi nama baru yang lebih jelas (misalnya, `rating_x` menjadi `user_rating`, `product_name_y` menjadi `product_name`) dan disimpan dalam DataFrame baru bernama `df_clean`.
   - Alasan: Langkah ini bertujuan untuk menyederhanakan dataset dan menghilangkan ambiguitas akibat adanya kolom dengan nama serupa setelah proses _merge_ (yang ditandai akhiran _x dan _y).
     
3. Rekayasa Fitur Kategori (_Feature Engineering_)
   - Proses: Tiga kolom kategori (`primary_category`, `secondary_category`, `tertiary_category`) digabungkan menjadi satu kolom tunggal bernama category. Setelah itu, ketiga kolom asli tersebut dihapus.
   - Alasan: Hal ini dilakukan untuk menciptakan satu fitur kategori yang komprehensif untuk setiap produk, yang sangat penting untuk model Content-Based dan evaluasinya.
     
4. Penanganan Data Duplikat
   - Proses: Duplikasi data diidentifikasi berdasarkan kombinasi `author_id` dan `product_id`. Metode `drop_duplicates()` digunakan untuk menghapus baris-baris duplikat, dengan hanya menyimpan ulasan pertama (keep='first') dari setiap pengguna untuk setiap produk.
   - Alasan: Langkah ini krusial untuk memastikan setiap interaksi pengguna-produk bersifat unik. Tanpa ini, opini seorang pengguna bisa terhitung beberapa kali untuk produk yang sama, yang dapat menyebabkan bias pada model Collaborative Filtering.
     
5. Penanganan Nilai Hilang (Missing Values)
   - Proses: Dilakukan dua strategi. Pertama, baris di mana nilai pada kolom `product_id`, `author_id`, atau `user_rating` kosong akan dihapus menggunakan `dropna()`. Kedua, nilai kosong pada kolom ingredients diisi dengan string kosong ('') menggunakan `fillna('')`.
   - Alasan: Data tanpa ID pengguna, ID produk, atau rating tidak dapat digunakan untuk model Collaborative Filtering, sehingga harus dihapus. Sementara itu, kolom ingredients diisi string kosong (bukan dihapus barisnya) agar tidak kehilangan data rating yang berharga, dan agar tidak terjadi error saat proses pengolahan teks.
     
6. Penanganan Outlier
   - Proses: Metode Interquartile Range (IQR) diterapkan pada kolom `avg_product_rating` untuk mengidentifikasi dan menghapus nilai-nilai ekstrem. Batas atas (`Q3 + 1.5*IQR`) dan batas bawah (`Q1 - 1.5*IQR`) dihitung, dan semua data yang berada di luar rentang ini akan dibuang.
   - Alasan: Berdasarkan visualisasi _boxplot_, terlihat adanya _outlier_ pada rating rata-rata produk. Menghapus _outlier_ membantu menciptakan dataset yang lebih stabil secara statistik dan mencegah nilai-nilai ekstrem tersebut mendistorsi proses pelatihan model.
     
7. Pembuatan `unique_products_df`
   - Membuat sebuah DataFrame baru yang hanya berisi satu baris untuk setiap produk unik. Hal ini dilakukan untuk efisiensi saat membangun model Content-Based dan Popularity-Based.
     
8. Pembuatan Fitur `soup`
   - Untuk model Content-Based, dibuat sebuah kolom baru bernama soup yang merupakan gabungan dari semua informasi teks produk (`product_name`, `brand_name`, `category`, `ingredients`). Hal ini dilakukan agar TF-IDF dapat menangkap esensi konten dari setiap produk secara holistik.
   ```python
   def create_soup(x):
       return x['product_name'] + ' ' + x['brand_name'] + ' ' + x['category'] + ' ' + x['ingredients']
   unique_products_df['soup'] = unique_products_df.apply(create_soup, axis=1)
   ```
   
9. Encoding dan Normalisasi untuk Deep Learning
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

Berikut adalah contoh _output_ dari sistem gabungan yang menampilkan rekomendasi dari ketiga model:

![image](https://github.com/user-attachments/assets/b21b90fc-4e61-4db0-aaaf-df39ffc03d87)

## Evaluation
1. **Evaluasi Model Content-Based Filtering (Precision@K)**

   Untuk mengukur performa model Content-Based yang menghasilkan daftar peringkat (ranking), metrik yang sesuai adalah Precision@K. Metrik ini ideal karena dapat menilai seberapa relevan item-item yang berada di posisi teratas dalam daftar rekomendasi. Precision@K mengukur proporsi item yang relevan dari K item teratas yang direkomendasikan. Formulanya adalah sebagai berikut:
   
   $$ \text{Precision@K} = \frac{\text{Jumlah item relevan di top-K}}{\text{K}} $$

   Proses evaluasi dilakukan dengan mengambil sampel acak sebanyak 100 produk dari dataset. Untuk setiap produk, dihasilkan 5 rekomendasi teratas (sehingga K=5) dan menghitung nilai Precision@5-nya. Setelah menerapkan pembobotan fitur pada model Content-Based untuk meningkatkan akurasinya, evaluasi menghasilkan skor yang jauh lebih baik. Berdasarkan pengujian pada 100 sampel acak, model ini mencapai rata-rata Precision@5 sebesar 81.20%. Ini adalah hasil yang sangat baik dan menunjukkan tingkat relevansi yang tinggi. Artinya, secara rata-rata, dari 5 produk yang disarankan oleh model, lebih dari 4 di antaranya berasal dari kategori yang sama dengan produk asli.
   
3. **Evaluasi Model Collaborative Filtering (RMSE)**

   Metrik evaluasi utama yang diterapkan pada model Collaborative Filtering (RecommenderNet) adalah Root Mean Squared Error (RMSE) untuk mengukur seberapa akurat prediksi ratingnya.
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

**Visualisasi Evaluasi Model Collaboratie=ve Filtering:**

![image](https://github.com/user-attachments/assets/2347bee1-701b-4a0f-9785-8161597e84f3)


Berdasarkan hasil pelatihan model `RecommenderNet`, grafik di atas menunjukkan kemajuan yang sangat baik. Terlihat bahwa nilai error pada data validasi (val_root_mean_squared_error) terus menurun secara konsisten dan mulai mendatar, yang menandakan model mencapai performa puncaknya. Titik terendah pada grafik ini dicapai di sekitar epoch ke-9 dengan nilai RMSE sekitar 0.236. Penting untuk diingat bahwa nilai RMSE ini dihitung pada rating yang telah dinormalisasi ke rentang [0, 1]. Untuk menginterpretasikannya dalam skala rating asli (1-5 bintang), perlu dikembalikan menggunakan rumus: 

$RMSE_{orig} = RMSE_{norm} \times (R_{max} - R_{min})$

$RMSE_{\text{asli}} = 0.236 \times (5 - 1) = 0.944$

Hasil ini menunjukkan bahwa, secara rata-rata, prediksi rating yang diberikan oleh model `RecommenderNet` memiliki kesalahan (deviasi) hanya sekitar 0.94 bintang dari rating yang sebenarnya diberikan oleh pengguna. Ini adalah hasil yang solid dan membuktikan bahwa model Collaborative Filtering mampu memberikan prediksi yang akurat secara kuantitatif. Penggunaan callback seperti EarlyStopping dan ModelCheckpoint sangat penting untuk memastikan model yang disimpan adalah versi dari epoch dengan performa terbaik.

## Referensi
[1] Statista. (2024). Beauty and Personal Care - Worldwide. Diakses pada 22 Juni 2025, dari https://www.statista.com/outlook/cmo/beauty-personal-care/worldwide
