"""Riset lanjutan terpisah dari menu eksperimen tim."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / 'eksperimen' / 'pipeline'
for p in (ROOT, PIPELINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

VERSION = 'v4.2'
