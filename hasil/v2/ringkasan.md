# More Than a Hack · Eksperimen malam

Seluruh angka di bawah berasal dari pertandingan Bundesliga IDSSE yang berbeda dari laga lomba.
Bobot skenario asli, celah pendek, celah panjang, dan gangguan kamera adalah 10%, 15%, 35%, dan 40%.

- Kandidat hasil seleksi dev: `geometry_2_4_+0`.
- Dev, spellF1 / macroF1: v1 0.3300 / 0.5875, kandidat 0.5033 / 0.8861.
- Holdout, spellF1 / macroF1: v1 0.3366 / 0.5787, kandidat 0.5364 / 0.8955.
- Stres tambahan, spellF1 / macroF1: v1 0.3056 / 0.5502, kandidat 0.5162 / 0.8835.
- Pilihan untuk review: `geometry_2_4_+0`.
- Rincian per pertandingan dan skenario ada di CSV agregat.
- Lima menit ground truth publik dipisahkan dari pelatihan, pemilihan, dan evaluasi kandidat.
- Batas eksekusi: 2026-11-10T06:00:00+07:00.

[IDSSE (Bassek et al., 2025)](https://doi.org/10.1038/s41597-025-04505-y)

- Validator, cek indeks publik, dan dua run identik. SHA-256 `d558bf063e5dec5dd2a4aced6b03287f51f748469f73dd12bbe15e7517e7716d`.
- CSV, dokumen teknis, dan ZIP kode ada di `kandidat_review/`.
