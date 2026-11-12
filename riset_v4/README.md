# Riset v4 penguasaan bola

Paket penelitian terpisah dari menu eksperimen tim 1–15. Jalankan perintah dari akar repository. Folder kerja Brian: `09_Riset_V4`. Hasil ini adalah kandidat penelitian, bukan pengganti submission otomatis.

## Yang dikerjakan

| Prioritas | Implementasi | Pemeriksaan |
|---|---|---|
| 1 | Model tujuh kelas: bukan batas + enam arah pergantian H/A/D. Kalibrasi suhu status dan sigmoid batas memakai pertandingan terpisah. | NLL status, Brier batas, prevalensi kejadian, probabilitas rata-rata. |
| 2 | Semua 110 fitur v3 dipertahankan; ditambah 93 fitur rapat, termasuk kecepatan relatif, gerak mendekat/menjauh, kesamaan gerak, dan kontinuitas kontak tim. | Simetri H/A, NaN, kecepatan tidak dihitung menyeberangi observasi hilang/lompatan slot. |
| 3 | Bobot episode pendek dibatasi, ditambah bobot dekat batas; sampel sekitar batas lebih rapat. | Recall episode <2 detik, 2–5 detik, dan ≥5 detik pada IoU 0,5. |
| 4 | Kandidat batas dicari di seluruh urutan; Viterbi langsung pada 25 fps. | Episode baru bisa terbentuk, bukan hanya menggeser batas lama. |
| 5 | TCN residual 48 kanal, dilasi 1/2/4/8/16/32, keluaran status + enam batas. | Potongan inferensi memakai halo 126 frame; hasil cocok dengan urutan penuh. |
| 6 | Optical flow dari kedua ujung celah, pemeriksaan bolak-balik, kemiripan patch, homografi, dan batas kecepatan. | Uji koordinat yang sengaja disembunyikan; celah meragukan ditolak; sidecar tidak mengubah tracking asli. |

## Menjalankan

Python 3.12. Instal dependensi dalam virtual environment khusus:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r riset_v4\requirements.txt
.\.venv\Scripts\python.exe -m unittest riset_v4.test_v4 -v
```

Pada AATA, environment khusus sudah disiapkan untuk model pohon dan video. PyTorch diperlukan untuk melatih/membaca TCN; pengujian TCN dalam pekerjaan ini dilakukan dengan CPU di lingkungan terpisah. Untuk GPU, instal build PyTorch yang sesuai driver. Kode memilih CUDA hanya bila tersedia.

Dengan cache pertandingan Brian:

```powershell
.\riset_v4\JALANKAN_RISET.ps1 -Mode benchmark
```

Tanpa cache lama:

```powershell
.\.venv\Scripts\python.exe -m riset_v4.fetch_public --output external_idsse
.\.venv\Scripts\python.exe -m riset_v4.run --cache external_idsse\cache --output outputs\riset_v4 --folds 0 2 --epochs 5 --steps 120
```

Output utama: `ringkasan_performa.csv`, `performa_detail.csv`, `manifest.json`, `status.json`, serta per fold model, kalibrasi, split, dan prediksi. Kode/data/konfigurasi berbeda harus memakai folder output baru. Proses dapat dilanjutkan dengan perintah yang persis sama; pertandingan yang sudah selesai dilewati. Model disimpan setelah seluruh pelatihan dan kalibrasi fold selesai.

Untuk lima fold, gunakan `--folds 0 1 2 3 4`. Ini memperluas evaluasi silang, bukan menambahkan pertandingan independen. Fitur besar dipetakan dari disk; sediakan beberapa GB ruang kosong. Jangan menjalankan banyak benchmark serentak pada laptop dengan RAM terbatas.

## Protokol

Lima pertandingan pengembangan: WMX, WN1, WOH, WOY, WPY (awalan `J03`). Fold k: pertandingan k untuk uji, k+1 untuk kalibrasi, tiga lainnya untuk pelatihan. WQQ dan WR9 diuji kembali pada tiap fold. Ketujuh pertandingan telah dipakai dalam riset sebelumnya; **tidak ada klaim holdout baru**.

Latihan default memakai original/camera_gaps/drift; uji memakai original, short_gaps, long_gaps, camera_gaps, drift. Bobot laporan mengikuti protokol lama: 0,10/0,15/0,30/0,30/0,15. Baseline `legacy_refit` dilatih ulang dengan pertandingan, skenario, dan 90 iterasi yang sama; baseline ini bukan angka lama 0,5148 dari protokol berbeda. Pertandingan kalibrasi tidak masuk fit model. Suhu/probabilitas dan enam kombinasi decoder hanya dipilih pada pertandingan kalibrasi. TCN dilatih 5 epoch × 120 langkah; ini anggaran awal yang eksplisit, bukan pencarian hiperparameter tuntas.

Scorer berasal dari `evaluation/possession_metrics.py`. MacroF1 tetap dilaporkan, tetapi pemilihan dibandingkan pada spellF1. Nilai antar-skenario dirata-rata per pertandingan terlebih dahulu; pengulangan seed/fold bukan pertandingan tambahan. Ensemble 50:50 ditentukan sebelumnya, tidak dicari pada skor uji.

Tidak ada label publik/privat BAR–ATH yang dibaca untuk melatih atau memilih model. Tidak ada override keras Smart Tagging baru pada inferensi. Skor eksternal tidak sama dengan skor target kompetisi.

## Video dan inferensi

```powershell
.\.venv\Scripts\python.exe -m riset_v4.video --data-dir 'D:\dataset' --video 'D:\dataset\tactical\BAR-ATH_tactical.mp4' --output outputs\video --top 40 --selftest 20
.\.venv\Scripts\python.exe -m riset_v4.predict --model outputs\riset_v4\fold_0\tree_candidate.joblib --data-dir 'D:\dataset' --output outputs\candidate.csv
```

Tambahkan `--recovery outputs\video\ball_recovery.npz` untuk membandingkan inferensi dengan pemulihan video. Sidecar harus berasal dari tracking yang sama. `candidate.probabilities.npz` dapat diberikan ke video melalui `--probabilities` agar prioritas celah memasukkan ketidakpastian model.

Video wajib rekaman taktis 199.319 frame pada 25 fps. Homografi memetakan koordinat lapangan ternormalisasi ke piksel. Video broadcast memiliki timeline lain dan tidak diterima oleh jalur ini. Default hanya mencoba celah ≤250 frame dengan kedua jangkar tersedia. Celah lebih panjang, bola tertutup, kamera tidak melihat bola, atau jangkar buruk tetap hilang dan tercatat dalam `gap_audit.csv`. Mengisi angka tanpa bukti piksel akan merusak fitur kontak.

Uji video menyembunyikan koordinat tracking yang tersedia selama lima frame. Selisih yang dilaporkan adalah kesesuaian dengan tracking masukan, bukan label manusia independen dan bukan spellF1. Jangan mempromosikan hasil video hanya karena beberapa frame berhasil dipulihkan.

Data kompetisi, video, frame, dan sidecar tetap lokal. Kode tidak mengunggah aset pertandingan. Program juga tidak mengganti file submission atau paket 1–15 secara otomatis.
