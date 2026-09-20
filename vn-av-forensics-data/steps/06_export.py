from _run import run
from settings import CANDIDATES, DATASET_VERSION, EXPORT, MANIFESTS

if __name__ == "__main__":
    run(
        [
            "export",
            "--root",
            CANDIDATES,
            "--review",
            MANIFESTS / "clips.csv",
            "--output",
            EXPORT,
            "--dataset-id",
            DATASET_VERSION,
        ]
    )
