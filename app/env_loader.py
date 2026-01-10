"""
Load .env from the repo root if present.
"""

from pathlib import Path
from dotenv import load_dotenv


_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
