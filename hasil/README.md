# Hasil eksperimen

Seluruh skor di sini berasal dari pertandingan eksternal IDSSE. Skor privat Barcelona vs Athletic belum tersedia.

| Putaran | Isi | Keputusan |
|---|---|---|
| [v1](v1/idsse_scoreboard.csv) | Aturan geometri pada dua pertandingan pembanding | Arsip awal |
| [v2](v2/ringkasan.md) | Model geometri, pemisahan laga, dan uji gangguan | Menjadi acuan v3 |
| [v3](v3/ringkasan.md) | Model flat, two-head, campuran, dan variasi decoder | Pilih flat/play_bias |
| [Transisi](transisi/ringkasan.md) | Konteks, pembelajaran pergantian, dan celah panjang | Pertahankan flat/play_bias |

Rincian angka dan pembagian laga ada di CSV, decision.json, dan protocol.json masing-masing putaran. Tabel antarputaran memakai pembagian laga dan bobot skenario berbeda; bandingkan metode pada baris evaluasi yang sama.

## Validasi paket

Menu 1–15 menjalankan inferensi dengan empat baseline, satu heuristik, dan sepuluh konfigurasi model/decoder. Pilihan model memakai bobot yang sudah dilatih; menjalankan nomor berbeda tidak otomatis melatih ulang model.

CSV prediksi setiap run disimpan lokal di outputs/, bersama run.json yang memuat validator dan SHA-256. Hasil per frame dan aset lomba tidak disalin ke repo. Ringkasan pemeriksaan paket ada di [verifikasi_paket.json](verifikasi_paket.json).

## Batas kesimpulan

J03WQQ dan J03WR9 belum digunakan saat pemilihan awal v3, tetapi sudah pernah dilihat pada retest transisi. Retest itu tidak dihitung sebagai bukti dari pertandingan baru. Uji stres memakai mask berulang pada laga yang sama; seed bukan jumlah pertandingan independen. Pemeriksaan SkillCorner pada putaran transisi gagal menyelaraskan empat laga, sehingga tidak menghasilkan skor evaluasi yang sah.
