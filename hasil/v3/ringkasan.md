# Eksperimen lanjutan · Hijrah ke Barcelona

Kandidat yang dibekukan sebelum membuka holdout: `flat/play_bias`.
Selisih spellF1 pada pengembangan: +0.0068.
Seluruh skor berasal dari pertandingan eksternal IDSSE.

| Evaluasi | v2 spellF1 | Kandidat spellF1 | v2 macroF1 | Kandidat macroF1 |
|---|---:|---:|---:|---:|
| Holdout baru | 0.5579 | 0.5803 | 0.8936 | 0.9062 |
| Uji stres | 0.5409 | 0.5629 | 0.8828 | 0.8992 |

Holdout baru: J03WQQ dan J03WR9.
Uji stres memakai mask berulang pada dua laga yang sama. Seed bukan pertandingan independen.
Keputusan: promosikan kandidat untuk review.

- Verifikasi berkas kandidat lanjutan lolos validator dan cek indeks. SHA-256 `f62698f078d5dc917707662d0a9aadc471ae52c384a58de42a1bbcb1febf656e`.
- Kasus grid diagnostik: 192/192. Hasil grid tidak dipakai untuk memilih kandidat.
