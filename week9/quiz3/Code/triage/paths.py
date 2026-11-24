"""Resolve the shared inputs from the checked-out Week 9 directory."""
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
WEEK_ROOT = CODE_ROOT.parents[1]
ROUTING_DATA = WEEK_ROOT / 'quiz2' / 'Data'
CHAT_DATA = WEEK_ROOT / 'quiz3' / 'Data'
IMAGES = CHAT_DATA / 'images'
