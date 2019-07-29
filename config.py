from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent

DATA_DIR = ROOT_DIR / 'data'
assert DATA_DIR.exists()

TESTS_DIR = DATA_DIR / 'tests'
TESTS_DIR.mkdir(exist_ok=True)
