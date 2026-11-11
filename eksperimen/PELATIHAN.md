# Pelatihan dan evaluasi eksternal

Menu run.py menggunakan bobot beku. Untuk mengulang pelatihan, ambil dataset IDSSE dari sumber resmi yang ditautkan dalam dokumen teknis, lalu letakkan arsip/XML pertandingan di folder eksternal. Pipeline tidak mengunduh data pertandingan lomba.

Dari root repo, aktifkan environment Python 3.12 dan pasang requirements.txt. Tambahkan root repo dan eksperimen/pipeline ke PYTHONPATH agar impor modul menggunakan sumber paket ini.

```powershell
$env:PYTHONPATH = "$((Get-Location).Path);$((Get-Location).Path)\eksperimen\pipeline"
New-Item -ItemType Directory -Force outputs\latih_v2 | Out-Null
python eksperimen/pipeline/night_benchmark.py --output outputs/latih_v2 --idsse "D:\IDSSE" --deadline "2026-11-16T12:00:00+00:00"
New-Item -ItemType Directory -Force outputs\latih_v3 | Out-Null
python eksperimen/pipeline/explore_benchmark.py --root outputs/latih_v3 --external "D:\IDSSE" --old-cache outputs/latih_v2/cache_idsse_publik --reference outputs/latih_v2/model_idsse.joblib --deadline "2026-11-16T12:00:00+00:00" --stress-deadline "2026-11-16T18:00:00+00:00"
```

Ganti batas waktu dengan waktu di masa depan jika mengulang sesudah tanggal contoh. v2 memerlukan J03WMX, J03WN1, J03WOH, J03WOY, J03WPY; v3 menambahkan J03WQQ dan J03WR9. Lihat kode night_tracking.idsse_files untuk struktur berkas XML yang dikenali. Pelatihan jauh lebih lama daripada menjalankan inferensi menu.

Benchmark menulis keputusan dan tabel pada direktori baru. Mengulang benchmark tidak mengganti bobot dalam frozen/; bandingkan keputusan dan hasil dahulu. Riwayat putaran menggunakan kembali dua laga evaluasi pada retest transisi, sehingga laga tersebut sudah tidak independen untuk pemilihan baru.

Delapan uji aturan yang telah digunakan dapat dijalankan dari root repo:

```powershell
$env:PYTHONPATH = "$((Get-Location).Path);$((Get-Location).Path)\eksperimen\pipeline"
python -m unittest discover -s eksperimen/pipeline -p test_possession.py -v
```
