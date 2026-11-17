
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_ZN_DIR = ROOT_DIR / "raw_zn"
RAW_APO_DIR = ROOT_DIR / "raw_apo"
RAW_SUPPLEMENT_DIR = ROOT_DIR / "raw_supplement"

SCORED_MARKED_PATH = DATA_DIR / "CMM_validation_split_clustered.csv"

GRAPHS_PATH = DATA_DIR / "graphs.pt"
APO_GRAPHS_PATH = DATA_DIR / "graphs_apo_full.pt"

ESM_CACHE_PATH = DATA_DIR / "esm_features.pt"
APO_ESM_CACHE_PATH = DATA_DIR / "esm_features_apo.pt"
ESM_PCA_PATH = MODELS_DIR / "esm_pca.pt"

APO_MAPPING_PATH = DATA_DIR / "apo_uniprot_mapping.json"
APO_MANIFEST_PATH = DATA_DIR / "apo_manifest.json"


ESM_PCA_DIM = 32
ESM_MODEL_NAME = "esm2_t12_35M_UR50D"


LR = 1e-3
GRAD_CLIP = 1.0



MAX_CYCLES = 3
NOISE_STD = 1.5
WRONG_SITE_PROB = 0.15
EVAL_LOSS_WEIGHT = 0.3


VETO_WARMUP = 5
MAX_SITES = 4


CLUSTER_MODE = "donor"
CLUSTER_CUTOFF = 4.5
SITE_MATCH_RADIUS = 5.0
TRILATERATE_MIN_SIZE = 3


MAX_CA_ATOMS = 6000
MIN_CLUSTER_SIZE = 2
