"""Make `import src...` work when pytest is started from anywhere inside the repository."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
