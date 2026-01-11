"""
Load .env from the repo root if present.
"""

from pathlib import Path
from dotenv import load_dotenv


_HERE = Path(__file__).resolve().parent
_CANDIDATES = (_HERE / ".env", _HERE.parent / ".env")
for _path in _CANDIDATES:
    if _path.exists():
        load_dotenv(_path)
        break
