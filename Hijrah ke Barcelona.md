# Penguasaan Bola dari Tracking Tactical

**Hijrah ke Barcelona · More Than a Hack 2027 · Initial Challenge**  
**Kandidat saat ini: flat/play_bias · 11 November 2026**

## Pendekatan

Pipeline membaca tracking tactical dan Smart Tagging otomatis dari paket panitia. Posisi bola dan pemain dikonversi ke meter. Fitur mencakup jarak bola terhadap kedua tim, gerak bola, ketersediaan tracking, geometri kolektif pemain, serta konteks sebelum dan sesudah suatu waktu. Identitas track diperlakukan sebagai slot pelacakan, bukan identitas pemain tetap.

Model HistGradientBoostingClassifier multikelas memprediksi H, A, dan D. Prediksi dengan pertukaran kedua tim dirata-ratakan untuk mengurangi ketergantungan pada urutan tim. Model bekerja pada langkah lima frame; decoder temporal memilih urutan status lalu mengembangkannya kembali menjadi satu label tiap frame. Biaya pergantian H–A adalah 2, biaya pergantian dengan D adalah 4, dan bias D adalah −0,18. Konfigurasi serta bobot yang dipakai dibekukan di eksperimen/frozen/v3/.

ATD memakai frame_id mulai 1, sehingga indeks video adalah frame_id − 1. Smart Tagging memakai detik dari awal video tactical dan diubah dengan round(detik × 25). Kedua babak diproses terpisah memakai batas resmi games.py. Keluaran tepat 199.319 baris, frame 0–199318, dengan header frame,label. Di luar kedua babak dipakai X.

## Aset dan data latih

Inferensi memakai tiga aset tactical: metadata ATD XML, tracking ATD TXT, dan Smart Tagging XML. Video dapat dibuka lewat viewer lokal untuk memeriksa sinkronisasi. Video, homography, dan broadcast tidak menjadi input kandidat ini. Tidak ada koreksi label manual pada pertandingan lomba.

Model dilatih dari tracking dan label yang telah tersedia dalam dataset eksternal [IDSSE](https://doi.org/10.1038/s41597-025-04505-y), oleh Bassek dkk. (2025), berlisensi CC BY 4.0. [Berkas dataset](https://huggingface.co/datasets/pysport/idsse-data) harus diperoleh terpisah. Tim tidak membuat anotasi manual tambahan. Label status bola dan tim penguasa mengikuti anotasi penyedia dataset; pembacaan dan konversinya ada di night_tracking.py.

Lima laga pengembangan adalah J03WMX, J03WN1, J03WOH, J03WOY, dan J03WPY. Validasi pengembangan membagi laga menjadi dua fold, bukan membagi frame acak. Model final dilatih ulang pada lima laga tersebut. J03WQQ dan J03WR9 dipakai untuk evaluasi kandidat v3 sesudah konfigurasinya dibekukan. Metadata di decision.json merekam pembagian ini. Model mencakup fitur asli, celah bola pendek/panjang, gangguan tampilan kamera, dan pergeseran koordinat sintetis; label sintetis tetap berasal dari laga eksternal.

Lima menit ground truth publik lomba hanya untuk pemeriksaan format, arti label, dan sinkronisasi, bukan pelatihan atau pemilihan parameter. Ground truth privat lomba tidak digunakan.

## Aturan penguasaan dan restart

Urutan H/A/D mengikuti bukti fitur dan decoder temporal. Penalti transisi mempertahankan penguasa saat bukti lawan hanya singkat, termasuk ketika bola lepas. Biaya tersebut berlaku umum dan tidak terikat nomor frame tertentu. Bola yang hilang tidak langsung diberi D.

Setelah decoder, aturan tambahan bola mati memeriksa bukti restart dari Smart Tagging dan tracking. Tag set piece saja tidak cukup: sistem mencari bola relatif diam di lokasi yang sesuai dan dekat pemain, lalu gerak yang mendukung restart. Bukti bola keluar serta gol/kick-off diperiksa melalui lintasan bola. Implementasi lengkap ada di possession.py; aturan ini menghasilkan label secara otomatis.

## Hasil dan pemilihan

[Ringkasan v3](hasil/v3/ringkasan.md) dan CSV di folder yang sama membandingkan kandidat pada dua laga evaluasi serta uji stres. Kenaikan spellF1 kandidat terhadap v2 adalah sekitar 0,0224 pada evaluasi dua laga dan 0,0220 pada uji stres, dengan bobot skenario yang ditetapkan dalam kode. Kenaikan itu merupakan hasil IDSSE, bukan skor pertandingan lomba.

Pada putaran transisi berikutnya, rata-rata spellF1 pengembangan meningkat dari 0,5039 untuk flat/play_bias menjadi 0,5148 untuk switch. Namun, dua laga retest sudah pernah dipakai pada putaran v3, dan keempat laga SkillCorner gagal pemeriksaan sinkronisasi. Putaran tersebut mempertahankan flat/play_bias sesuai protokol pemeriksaannya. [Keputusan dan tabel lengkap](hasil/transisi/).

Seluruh 15 pilihan menu diperiksa dengan validator resmi. Nomor 7 mereproduksi kandidat saat ini. Bukti jumlah baris dan hash keluaran ada di hasil/verifikasi_paket.json. Hasil Barcelona vs Athletic dihasilkan lokal dan diserahkan melalui formulir resmi.

## Keterbatasan

Jumlah laga eksternal masih kecil dan penyedia tracking berbeda dari data lomba. Gangguan sintetis belum mewakili seluruh kesalahan kamera, deteksi, dan pelacakan sebenarnya. Celah bola panjang dapat menyembunyikan peralihan penguasaan. Penalti temporal dapat menghapus penguasaan singkat yang benar atau menggeser batas episode. Cabang restart Smart Tagging tidak memiliki anotasi eksternal dengan format yang sama. Metode ini memakai konteks masa depan sehingga sesuai untuk analisis setelah pertandingan, bukan inferensi langsung tanpa jeda. Skor privat pertandingan lomba belum diketahui.

## Reproduksi

Gunakan Python 3.12 dan dependensi yang dipatok di eksperimen/requirements-lock.txt. Dari root repo:

```powershell
python -m pip install -r requirements.txt
python eksperimen/run.py --method 7 --data-dir "D:\\MTAH\\data"
```

Di Windows, JALANKAN.bat menyiapkan environment lalu menampilkan menu. Folder data harus berisi tactical/ dengan nama berkas asli panitia. Setiap run membuat subfolder baru dalam outputs/ berisi Hijrah ke Barcelona.csv dan run.json; SHA-256 dapat dibandingkan dengan catatan verifikasi. Pemisahan output mencegah hasil anggota lain tertimpa.

Kode pelatihan awal dan eksplorasi tersedia di eksperimen/pipeline/night_benchmark.py serta explore_benchmark.py. Petunjuk reproduksi pelatihan ada di [eksperimen/PELATIHAN.md](eksperimen/PELATIHAN.md). Menu 1–15 menjalankan inferensi; bobot final sudah disertakan.

[Repo tim](https://github.com/aata4brian/more-than-a-hack) · [Aturan panitia](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
