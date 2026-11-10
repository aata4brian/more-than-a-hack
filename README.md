# Hijrah ke Barcelona

Ruang kerja tim untuk fase Ball Possession di More Than a Hack 2027. Setiap percobaan menghasilkan CSV `frame,label` dengan 199.319 baris untuk kamera tactical. Nomor **07** mereproduksi kandidat `flat/play_bias` yang sudah diverifikasi.

## Jalankan percobaan

1. Instal **Python 3.12** dengan Python Launcher di Windows. Ambil paket data dari tautan resmi panitia yang diterima anggota terdaftar.
2. Ekstrak paket dengan susunan `data/tactical/` dan `data/broadcast/`. Kalau data disimpan di folder lain, pilih folder induk `tactical/` ketika diminta.
3. Buka **`eksperimen/JALANKAN.bat`**, lalu ketik nomor **1–15**. Saat pertama dibuka, launcher memasang dependensi yang dipatok versinya.
4. Ambil `Hijrah ke Barcelona.csv` dan `run.json` dari subfolder baru di `outputs/`. Terminal akan memberi tahu apakah CSV lolos validator panitia.

Di terminal Windows, perintah yang sama bisa dijalankan dari root repo.

```powershell
.\eksperimen\JALANKAN.bat
.\eksperimen\JALANKAN.bat --method 7 --data-dir "D:\MTAH\data"
```

Untuk Python 3.12 dengan lingkungan virtual yang sudah aktif:

```bash
python -m pip install -r eksperimen/requirements-lock.txt
python eksperimen/run.py --list
python eksperimen/run.py --method 7 --data-dir /path/to/data
```

## Pembagian lima anggota

| Anggota | Nomor |
|---|---|
| A | 1, 6, 11 |
| B | 2, 7, 12 |
| C | 3, 8, 13 |
| D | 4, 9, 14 |
| E | 5, 10, 15 |

Keempat baseline panitia, dua versi tim terdahulu, kandidat utama, dan delapan variasi model atau decoder tersedia dari menu yang sama. Rincian tiap nomor ada di [panduan eksperimen](eksperimen/README.md). Hasil eksternal yang sudah selesai ada di [hasil](hasil/README.md).

## Susunan repo

| Lokasi | Isi |
|---|---|
| [games.py](games.py) | Konstanta pertandingan dan pemetaan frame dari panitia |
| [starter/](starter/) | Contoh tracking, tagging, dan baseline panitia |
| [evaluation/](evaluation/) | Scorer dan validator panitia |
| [data/](data/) | Petunjuk lokasi paket resmi di komputer anggota |
| [eksperimen/](eksperimen/) | Menu 1–15, kode pipeline, dan bobot dari data publik IDSSE |
| [hasil/](hasil/) | Ringkasan dan tabel evaluasi pertandingan eksternal |
| [viewer/](viewer/) | Viewer HTML untuk memeriksa sinkronisasi video lokal dan prediksi |
| `outputs/` | CSV prediksi dan catatan run lokal |

Panduan resmi [CHALLENGE](CHALLENGE.md), [DATA](DATA.md), [EVALUATION](EVALUATION.md), [SUBMISSION](SUBMISSION.md), dan [RULES](RULES.md) disalin dari [repo panitia](https://github.com/metrica-sports/more-than-a-hack-2027/tree/f8f2629ff1e0985f81d7e59d8413678514ee1530).

Video, tracking, homography, dan tagging Barcelona vs Athletic berasal dari paket berlisensi panitia. Setiap anggota membuka berkas itu di komputernya sendiri. Hasil per frame untuk pengumpulan disimpan di `outputs/` dan diserahkan lewat formulir panitia. Lima menit label publik dipakai untuk pengecekan format, indeks, dan sinkronisasi. Pemilihan metode memakai pertandingan eksternal.
