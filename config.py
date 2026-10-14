
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


CSV_DIR = DATA_DIR /"csv"
RAW_BENCHMARK_DIR = ROOT_DIR / "raw_supplement"
HOLO_SET_DIR = DATA_DIR / "HOLO_SET"
DATA_MANAGMENT_DIR = SRC_DIR/"data_managment"


GRAPHS_PATH = DATA_DIR / "graphs.pt"
APO_GRAPHS_PATH = DATA_DIR / "graphs_apo_full.pt"

ESM_CACHE_PATH = DATA_DIR / "esm_features.pt"
APO_ESM_CACHE_PATH = DATA_DIR / "esm_features_apo.pt"

APO_MAPPING_PATH = DATA_DIR / "apo_uniprot_mapping.json"
APO_MANIFEST_PATH = DATA_DIR / "apo_manifest.json"