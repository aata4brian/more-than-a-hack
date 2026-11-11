# Review video lokal

Buka **BUKA_VIEWER.bat** dari folder utama paket. Viewer memuat video tactical, CSV prediksi terbaru di outputs/, referensi publik 5 menit, dan indikator bola dari ATD. Browser tidak mengirim berkas ke layanan luar. Terminal harus tetap terbuka selama menonton; tekan Ctrl+C untuk menutup server.

Lokasi data dipilih dengan --data-dir atau variabel MTAH_DATA_DIR. Konfigurasi pribadi dapat ditulis dalam .mtah-local.json di folder utama. File ini diabaikan Git dan tidak masuk ZIP tim.

```json
{"data_dir": "D:/MTAH/data"}
```

Pilih CSV tertentu dengan:

```powershell
.\BUKA_VIEWER.bat --data-dir "D:\MTAH\data" --prediction "D:\hasil\Hijrah ke Barcelona.csv"
```

Kolom prediction opsional dalam konfigurasi lokal dapat mengunci CSV yang dibuka. Tanpa kolom itu, viewer memilih hasil terbaru di outputs/. Jika belum ada prediksi, video dan referensi publik tetap dapat dibuka; pilih CSV manual atau jalankan satu eksperimen dahulu.

Tombol **Cuplikan publik** melompat ke bagian berlabel resmi. Tombol panah berpindah per frame; **Ganti label** menuju pergantian prediksi berikutnya. Referensi di luar 5 menit ditampilkan sebagai tidak tersedia. Video broadcast punya timeline berbeda dan tidak boleh dipakai sebagai pengganti tactical untuk overlay ini. Untuk menonton broadcast, buka MP4 broadcast langsung dari folder data.

Drive untuk desktop harus aktif bila data berada di drive streaming. Jika pemuatan lambat, gunakan menu Drive **Available offline** pada perangkat masing-masing. Video tidak disalin ke paket sumber dan tidak dimasukkan GitHub.

Viewer hanya untuk memeriksa pipeline dan sinkronisasi. Aturan panitia tetap melarang anotasi manual, koreksi prediksi per frame, atau aturan khusus dari kejadian pertandingan.
