import sys
from pathlib import Path

# Ensure the repo root is on sys.path so tests can import the top-level `workflow/`
# namespace package (and other repo-local modules) regardless of pytest import mode.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
