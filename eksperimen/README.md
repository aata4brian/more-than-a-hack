# Menu eksperimen

Jalankan `JALANKAN.bat` dari Windows atau `python run.py` dari lingkungan Python 3.12 yang sudah memasang `requirements-lock.txt`. `--method` bisa diulang jika ingin menjalankan lebih dari satu nomor sekaligus. Semua nomor menghasilkan CSV di folder berbeda.

| Nomor | Metode | Yang diuji |
|---|---|---|
| 1 | Baseline raw | Tim pemain terdekat pada tiap frame |
| 2 | Baseline ±25 | Voting dengan jendela pendek |
| 3 | Baseline ±50 | Baseline acuan panitia |
| 4 | Baseline ±125 | Voting dengan jendela lebih panjang |
| 5 | Heuristik v1 | Aturan jarak, gerak, tracking, dan restart |
| 6 | Geometry v2 | Klasifikasi dengan geometri tim dan decoder |
| 7 | Flat / play bias | Kandidat terverifikasi, model multikelas dan bias bola hidup |
| 8 | Flat / standard | Decoder tanpa tambahan bias bola hidup |
| 9 | Flat / fast | Biaya pergantian yang lebih rendah |
| 10 | Flat / stable | Biaya pergantian yang lebih tinggi |
| 11 | Two-head / play bias | Status bola dan pemilik bola diprediksi secara terpisah |
| 12 | Blend / play bias | Rata-rata probabilitas flat dan two-head |
| 13 | Blend v2 / play bias | Gabungan probabilitas v2, flat, dan two-head |
| 14 | Context | Biaya pergantian ikut berubah sesuai dukungan sebelum dan sesudah batas |
| 15 | Context gap | Perubahan biaya lebih besar saat posisi bola hilang |

Pilihan 1–4 memakai baseline resmi. Pilihan 5–15 memakai tracking tactical dan petunjuk restart yang disediakan panitia. Model beku pada 6–15 dilatih dari pertandingan [IDSSE](https://doi.org/10.1038/s41597-025-04505-y), dengan rincian pembagian laga di `frozen/`.

Setiap hasil memuat catatan nomor eksperimen, jumlah baris, dan SHA-256 di `run.json`. Untuk membandingkan skor, gunakan tabel evaluasi eksternal dan periksa bahwa dua metode diuji pada pertandingan serta skenario yang sama. Hasil Barcelona vs Athletic yang terlihat di viewer dipakai untuk debugging sinkronisasi dan format.
