# Pilah.in

Aplikasi web berbasis AI untuk membantu warga memilah sampah. Foto sampahmu, lalu aplikasi menebak jenisnya (kertas, plastik, logam, atau kaca) dan memberi saran penanganan.

**Coba langsung:** https://ny7obqtephnsytne6brepp.streamlit.app

> Catatan: aplikasi di Streamlit gratis bisa "tertidur" kalau lama tidak dibuka. Kalau muncul tulisan "waking up", tunggu satu-dua menit.

## Kenapa proyek ini ada

Tempat pembuangan akhir seperti Bantargebang menerima ribuan ton sampah per hari, dan banyak rumah tangga belum memilah sampah dari sumbernya. Sampah yang sudah terpilah jauh lebih mudah didaur ulang. Pilah.in adalah percobaan kecil untuk membuat langkah pertama itu lebih mudah.

## Cara kerja

1. Pengguna mengunggah foto atau memotret sampah.
2. Model klasifikasi gambar (MobileNetV2, transfer learning) memprediksi kelasnya.
3. Aplikasi menampilkan kelas terprediksi, tingkat keyakinan, dan saran penanganan.
4. Kalau keyakinan model rendah (di bawah 60%), aplikasi memberi peringatan, bukan menebak sembarangan.

## Hasil

Model dilatih dari dataset TrashNet (sekitar 2.390 gambar, 80% latih dan 20% validasi) dengan 4 kelas.

- **Akurasi validasi: 92,9%**

| Kelas | Benar dikenali |
|---|---|
| kertas | 98% |
| logam | 95% |
| plastik | 90% |
| kaca | 84% |

Kesalahan paling sering: botol kaca ditebak sebagai plastik atau logam.

## Keterbatasan (penting)

- Foto TrashNet diambil dengan latar polos dan cahaya rapi. Akurasi pada foto dunia nyata (sampah menumpuk, latar berantakan) kemungkinan **lebih rendah** dari angka di atas, dan belum diuji secara sistematis.
- Hanya ada 4 kelas. Sampah organik, B3, dan jenis lain belum didukung.
- Saran penanganan bersifat umum. Aturan bank sampah di tiap daerah bisa berbeda.
- Model menilai tampilan benda, bukan kebersihannya.

## Rencana pengembangan

- Menambah kelas sampah organik.
- Menambah foto dunia nyata untuk melatih ulang model.
- Peta lokasi bank sampah terdekat.

## Menjalankan sendiri

Aplikasi:

    pip install -r requirements.txt
    streamlit run app.py

Melatih ulang model (butuh dataset TrashNet `dataset-resized` di folder yang sama):

    python train_pilahin.py

## Struktur repo

- `app.py` : aplikasi Streamlit
- `train_pilahin.py` : skrip pelatihan model
- `pilahin_mobilenetv2.pt` : bobot model hasil pelatihan
- `requirements.txt` : daftar library

## Kredit

- Dataset: TrashNet oleh Gary Thung dan Mindy Yang
- Model dasar: MobileNetV2 (torchvision)
- Dibuat oleh Farel [![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/farelandreassitohang)
[![TikTok](https://img.shields.io/badge/TikTok-000000?style=for-the-badge&logo=tiktok&logoColor=white)](https://tiktok.com/@FAREL ANDREAS)

Masukan sangat diterima, terutama foto sampah dunia nyata yang salah ditebak.
