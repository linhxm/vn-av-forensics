from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_VERSION = "dataset_v002"
DOWNLOAD_RUN = "download_001"
CUT_RUN = "cut_001"
SOURCES = ROOT / "data/sources" / DATASET_VERSION
RAW = ROOT / "data/raw" / DATASET_VERSION / DOWNLOAD_RUN
CANDIDATES = ROOT / "data/candidates" / DATASET_VERSION
CUT = CANDIDATES / CUT_RUN
MANIFESTS = ROOT / "data/manifests" / DATASET_VERSION
EXPORT = ROOT / "exports" / DATASET_VERSION
