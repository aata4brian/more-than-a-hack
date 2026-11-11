# Eksperimen penguasaan bola

Pengembangan memakai lima laga IDSSE dalam dua fold terpisah per laga.
Baseline flat/play_bias mendapat spellF1 0.5039 dan macroF1 0.8822 dengan bobot skenario yang dideklarasikan.

| Metode | spellF1 | macroF1 | Menang per laga |
|---|---:|---:|---:|
| switch | 0.5148 | 0.8843 | 5/5 |
| gap_switch | 0.5139 | 0.8853 | 5/5 |
| switch_gap | 0.5113 | 0.8839 | 5/5 |
| gap_context | 0.5067 | 0.8837 | 4/5 |
| gap_fit | 0.5064 | 0.8837 | 4/5 |
| gap_blend | 0.5049 | 0.8833 | 3/5 |
| context_gap | 0.5046 | 0.8825 | 4/5 |
| context | 0.5043 | 0.8824 | 4/5 |

Terbaik di pengembangan adalah switch (+0.0109).
Kandidat yang lolos ambang pengembangan: switch.
Retest IDSSE J03WQQ berubah +0.0112 spellF1. Laga ini pernah dipakai pada putaran sebelumnya.
Retest IDSSE J03WR9 berubah +0.0079 spellF1. Laga ini pernah dipakai pada putaran sebelumnya.
SkillCorner menghasilkan 0/4 laga yang lolos pemeriksaan, di bawah minimum tiga laga.
Keputusan akhir: pertahankan flat/play_bias.
Skor privat laga Barcelona vs Athletic belum diketahui.
