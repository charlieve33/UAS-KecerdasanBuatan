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

