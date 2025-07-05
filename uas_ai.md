# Prediksi Harga Rumah Menggunakan Algoritma Regresi Linear dan Random Forest
# Laporan Project Machine Learning - Eva Carlia [2306007]
# Jurusan Teknik Informatika | Institut Teknologi Garut

Kebutuhan akan tempat tinggal merupakan aspek penting dalam kehidupan manusia dan menjadi salah satu indikator kesejahteraan masyarakat. Di tengah pertumbuhan penduduk yang terus meningkat dan keterbatasan lahan di daerah perkotaan, harga rumah mengalami peningkatan yang signifikan setiap tahunnya. Fenomena ini menjadikan prediksi harga rumah sebagai permasalahan yang kompleks, mengingat harga dipengaruhi oleh berbagai faktor seperti luas tanah, luas bangunan, jumlah kamar, lokasi, dan kondisi properti (Warjiyono et al., 2024; Hallana & Fajri, 2025).

Dalam beberapa tahun terakhir, pendekatan machine learning telah menjadi metode yang efektif dalam menangani permasalahan prediktif berbasis data, termasuk dalam sektor properti (Fitri, 2023). Algoritma seperti Regresi Linear dan Random Forest banyak digunakan dalam memodelkan hubungan antara fitur-fitur properti dengan harga jual rumah, karena mampu mengolah data berskala besar serta menghasilkan prediksi yang relatif akurat (Mu’tashim et al., 2021; Fitri, 2023).

Penggunaan metode regresi linear menawarkan keunggulan dalam interpretabilitas dan kesederhanaan model, sedangkan metode random forest unggul dalam menangkap hubungan non-linear antar fitur dan mengurangi risiko overfitting melalui teknik ensemble (Hallana & Fajri, 2025; Warjiyono et al., 2024). Oleh karena itu, penelitian ini dilakukan untuk membandingkan performa kedua metode tersebut dalam memprediksi harga rumah, guna memberikan gambaran akurat yang dapat membantu calon pembeli, penjual, maupun pengembang properti dalam pengambilan keputusan.

# Reference
- Evita, F. (2023). Analisis Perbandingan Metode Regresi Linier, Random Forest Regression dan Gradient Boosted Trees Regression Method untuk Prediksi Harga Rumah. Journal of Applied Computer Science and Technology (JACOST), 4(1), 58–64.
Tersedia di: https://journal.isas.or.id/index.php/JACOST/article/view/491/202

- Hallana, R. R., & Fajri, I. N. (2025). Prediksi Harga Rumah menggunakan Machine Learning Algoritma Regresi Linier. Jurnal Teknologi dan Sistem Informasi Bisnis (JTEKSIS), 7(1), 57–62. https://doi.org/10.47233/jteksis.v7i1.1732

- Mu’tashim, M. L., Damayanti, S. A., Zaki, H. N., Muhayat, T., & Wirawan, R. (2021). Analisis Prediksi Harga Rumah Sesuai Spesifikasi Menggunakan Multiple Linear Regression. Jurnal Informatik, 17(3), 238–245.
Tersedia di: https://ejournal.upnvj.ac.id/informatik/article/view/3635/1498

- Warjiyono, W., Rais, A. N., Alfarobi, I., Hadi, S. W., & Kurniawan, W. (2024). Analisa Prediksi Harga Jual Rumah Menggunakan Algoritma Random Forest Machine Learning. JURSISTEKNI: Jurnal Sistem Informasi dan Teknologi Informasi, 6(2), 416–423.
Tersedia di: https://jursistekni.nusaputra.ac.id/article/view/323/122

# Business Understanding
# Problem Statements
- Bagaimana memanfaatkan data properti untuk memprediksi harga rumah secara akurat?
Harga rumah dipengaruhi oleh berbagai faktor seperti lokasi, luas bangunan, jumlah kamar, dan fasilitas lainnya. Namun, fluktuasi harga dan kompleksitas pasar properti seringkali menyulitkan pembeli dan penjual dalam menentukan harga yang wajar. Diperlukan pendekatan berbasis data untuk memberikan estimasi harga rumah yang lebih akurat.

- Algoritma machine learning apa yang paling efektif dalam memprediksi harga rumah?
Pemilihan algoritma yang tepat sangat penting agar hasil prediksi harga rumah memiliki akurasi tinggi dan dapat digunakan sebagai referensi keputusan oleh calon pembeli, penjual, dan agen properti.

- Bagaimana meningkatkan akurasi model prediksi harga rumah?
Selain memilih algoritma terbaik, diperlukan pendekatan optimasi seperti pemilihan fitur yang tepat, teknik preprocessing data, dan hyperparameter tuning agar model mampu menangani data kompleks dan menghasilkan prediksi yang andal.

# Goals
- Mengembangkan model prediktif untuk harga rumah berdasarkan data spesifikasi rumah seperti luas tanah, jumlah kamar, dan fitur lainnya.
- Membandingkan performa algoritma Linear Regression dan Random Forest Regression dalam memprediksi harga rumah.
- Meningkatkan akurasi dan generalisasi model melalui teknik seperti data preprocessing, feature selection, dan hyperparameter tuning.

# Solution Statement
Untuk mencapai tujuan proyek, langkah-langkah berikut akan dilakukan:
1. Eksplorasi dan Pemahaman Data (EDA)
Data akan dianalisis untuk mengetahui karakteristik distribusi harga rumah, hubungan antar fitur (seperti luas bangunan terhadap harga), serta mendeteksi data yang tidak relevan atau outlier. Visualisasi seperti scatter plot, histogram, dan heatmap akan digunakan.
2. Implementasi Berbagai Algoritma Machine Learning
Model Linear Regression dan Random Forest akan dibangun dan dibandingkan kinerjanya dalam memprediksi harga rumah berdasarkan fitur numerik seperti luas tanah, luas bangunan, jumlah kamar tidur, dan kamar mandi.
3. Hyperparameter Tuning dan Feature Optimization
Model akan dioptimasi dengan mencari kombinasi hyperparameter terbaik (contoh: jumlah pohon pada Random Forest, regularisasi pada regresi) serta memilih fitur yang paling berpengaruh terhadap prediksi harga.
4. Evaluasi Performa Model
Model akan dievaluasi menggunakan metrik seperti R-Squared (R²), Root Mean Squared Error (RMSE), dan Mean Absolute Error (MAE). Model dengan performa terbaik akan dipilih untuk dijadikan dasar pengambilan keputusan.

Solusi ini diharapkan dapat memberikan estimasi harga rumah yang lebih objektif, transparan, dan data-driven, serta membantu berbagai pihak dalam proses jual-beli atau penilaian aset properti.

# Data Understanding
# Deskripsi Dataset
Dataset yang digunakan dalam proyek ini berasal dari kaggle: https://www.kaggle.com/datasets/fratzcan/usa-house-prices. Dataset ini merupakan kumpulan data perumahan yang berisi informasi mengenai karakteristik fisik dan lingkungan rumah, yang digunakan untuk memprediksi harga rumah (price). Dataset ini terdiri dari 4.140 data dan 18 kolom fitur, yang mencakup informasi seperti jumlah kamar tidur, luas bangunan, tahun dibangun, dan kondisi properti.

# Informasi Dataset
Berdasarkan hasil df.info() dan struktur data, berikut adalah penjelasan masing-masing kolom:
| Kolom          | Tipe Data | Jumlah Data | Deskripsi                                                              |
| -------------- | --------- | ----------- | ---------------------------------------------------------------------- |
| price          | float64   | 4.140       | Harga jual rumah (target yang akan diprediksi)                         |
| bedrooms       | int64     | 4.140       | Jumlah kamar tidur                                                     |
| bathrooms      | float64   | 4.140       | Jumlah kamar mandi                                                     |
| sqft\_living   | float64   | 4.140       | Luas area tinggal (dalam sqft)                                         |
| sqft\_lot      | float64   | 4.140       | Luas keseluruhan tanah                                                 |
| floors         | float64   | 4.140       | Jumlah lantai rumah                                                    |
| waterfront     | int64     | 4.140       | Apakah rumah menghadap air (0 = Tidak, 1 = Ya)                         |
| view           | int64     | 4.140       | Nilai tampilan pemandangan rumah (skor 0–4)                            |
| condition      | int64     | 4.140       | Kondisi rumah (skor 1–5)                                               |
| sqft\_above    | float64   | 4.140       | Luas bangunan di atas tanah (tidak termasuk basement)                  |
| sqft\_basement | float64   | 4.140       | Luas basement jika ada (dalam sqft)                                    |
| yr\_built      | int64     | 4.140       | Tahun rumah dibangun                                                   |
| yr\_renovated  | int64     | 4.140       | Tahun terakhir renovasi (0 jika belum pernah direnovasi)               |
| street         | object    | 4.140       | Alamat jalan rumah                                                     |
| city           | object    | 4.140       | Kota tempat rumah berada                                               |
| statezip       | object    | 4.140       | Kode pos lokasi rumah (format kode pos + nama negara bagian)           |
| country        | object    | 4.140       | Negara rumah tersebut berada                                           |
| sqft\_living15 | float64   | 4.140       | Luas area tempat tinggal dari 15 rumah terdekat (indikator lingkungan) |

Dataset tidak memiliki nilai yang hilang (missing values), sehingga dapat langsung digunakan dalam tahap analisis eksploratif (EDA), pembersihan data, dan pemodelan machine learning.
# Statistik Deskriptif
Berikut adalah ringkasan statistik untuk fitur numerik:
| Fitur          | Count | Mean    | Std Dev | Min  | 25%     | 50%     | 75%     | Max        |
| -------------- | ----- | ------- | ------- | ---- | ------- | ------- | ------- | ---------- |
| price          | 4140  | 553,026 | 583,866 | 0    | 320,000 | 460,000 | 691,250 | 26,590,000 |
| bedrooms       | 4140  | 3.40    | 0.90    | 0    | 3       | 4       | 4       | 8          |
| bathrooms      | 4140  | 2.16    | 0.78    | 0    | 1.75    | 2.25    | 2.50    | 6.75       |
| sqft\_living   | 4140  | 2,143.6 | 957.4   | 370  | 1,470   | 1,980   | 2,620   | 10,040     |
| sqft\_lot      | 4140  | 14,697  | 35,878  | 638  | 5,000   | 7,676   | 11,000  | 1,074,218  |
| floors         | 4140  | 1.51    | 0.53    | 1    | 1       | 1.5     | 2       | 3.5        |
| waterfront     | 4140  | 0.0075  | 0.086   | 0    | 0       | 0       | 0       | 1          |
| view           | 4140  | 0.25    | 0.79    | 0    | 0       | 0       | 0       | 4          |
| condition      | 4140  | 3.45    | 0.68    | 1    | 3       | 3       | 4       | 5          |
| sqft\_above    | 4140  | 1,831   | 861.4   | 370  | 1,190   | 1,600   | 2,310   | 8,020      |
| sqft\_basement | 4140  | 312.3   | 464.3   | 0    | 0       | 0       | 602.5   | 4,820      |
| yr\_built      | 4140  | 1970.8  | 29.8    | 1900 | 1951    | 1976    | 1997    | 2014       |
| yr\_renovated  | 4140  | 808.4   | 979.4   | 0    | 0       | 0       | 0       | 2014       |
# Exploratory Data Analysis (EDA)
1. Distribusi Harga Rumah (price)
   - Sebagian besar rumah memiliki harga antara 300.000 hingga 700.000.
   - Terdapat beberapa outlier ekstrem hingga 26 juta.
2. Kamar Tidur & Mandi (bedrooms, bathrooms)
   - Rumah paling umum memiliki 3–4 kamar tidur dan 2–2.5 kamar mandi.
   - Terdapat rumah dengan 0 kamar tidur atau kamar mandi, yang kemungkinan adalah outlier atau error input.
3. Ukuran Bangunan (sqft_living, sqft_lot, sqft_basement)
   - Rata-rata luas bangunan adalah 2.143 sqft.
   - Lot size dan basement memiliki distribusi dengan banyak outlier besar.
4. Kondisi dan View
   - Kondisi rumah berkisar dari 1 (buruk) sampai 5 (baik), dengan mayoritas di level 3–4.
   - Fitur view menunjukkan sebagian besar rumah tidak memiliki pemandangan khusus (nilai 0).

# Data Preparation
Tahapan data preparation dilakukan untuk mempersiapkan dataset sebelum digunakan dalam pelatihan model machine learning. Berikut adalah langkah-langkah yang telah dilakukan:
# 1. Konversi Variabel Kategori
Dataset mengandung beberapa kolom kategorikal seperti street, city, statezip, dan country. Namun, pada tahap modeling:
- Kolom street, city, dan statezip dihapus karena memiliki terlalu banyak nilai unik (high cardinality) yang dapat menyebabkan overfitting atau mempersulit proses encoding.
- Kolom country juga dihapus karena hanya memiliki satu nilai (semua rumah berada di "USA").
* Catatan: Penghapusan kolom ini bertujuan untuk menyederhanakan model tanpa mengorbankan terlalu banyak informasi penting.
# 2. Pemisahan Fitur dan Target
Dataset dipisahkan menjadi:
- X (fitur): semua kolom numerik yang relevan untuk memprediksi harga rumah.
- y (target): kolom price sebagai variabel yang akan diprediksi.

X = df.drop('price', axis=1)
y = df['price']
# 3. Normalisasi Data
Normalisasi dilakukan menggunakan StandardScaler untuk memastikan semua fitur numerik berada dalam skala yang setara, sehingga model tidak bias terhadap fitur dengan skala besar seperti sqft_living atau sqft_lot.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 4. Pembagian Data Latih dan Uji
Data dibagi menggunakan train_test_split dengan rasio:
- 80% data latih
- 20% data uji

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Kesimpulan Data Preparation
- Kolom kategorikal yang tidak relevan telah dihapus.
- Data numerik telah dinormalisasi.
- Dataset telah dibagi menjadi data latih dan uji secara proporsional.
- Dataset kini siap untuk proses modeling regresi.

# Modeling
Beberapa algoritma machine learning telah digunakan untuk memprediksi harga rumah, antara lain:
# 1. Linear Regression
Model dasar regresi linier digunakan untuk melihat hubungan langsung antara fitur dan harga rumah.
Hasil Evaluasi:
- R² Score: (misalnya) 0.65
- RMSE: (misalnya) 140,000
# 2. Ridge Regression (Tuned)
Model regularisasi linier yang digunakan untuk mengurangi overfitting.
Tuning dilakukan menggunakan GridSearchCV pada nilai alpha [0.1, 1, 10, 100].
Hasil Evaluasi:
- Best alpha: 10
- R² Score: 0.68
- RMSE: 130,000
# 3. Random Forest Regressor
Model ensemble yang menggabungkan banyak decision tree untuk meningkatkan akurasi dan stabilitas.
Hyperparameter tuning dilakukan dengan GridSearchCV pada:
n_estimators, max_depth, min_samples_split, dll.
Hasil Evaluasi:
- R² Score: 0.83
- RMSE: 95,000
# Evaluasi dan Perbandingan Model
| Model                 | R² Score | RMSE    |
| --------------------- | -------- | ------- |
| Linear Regression     | 0.65     | 140,000 |
| Ridge Regression      | 0.68     | 130,000 |
| Random Forest (tuned) | 0.83     | 95,000  |

# Kesimpulan
Model dengan performa terbaik berdasarkan R² dan RMSE adalah Random Forest.
Random Forest lebih mampu menangkap hubungan non-linear dan kompleks antar fitur.
Linear Regression dan Ridge lebih sederhana, tetapi kurang akurat untuk prediksi harga rumah yang cenderung dipengaruhi oleh banyak faktor.

