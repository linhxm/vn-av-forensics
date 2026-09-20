from _run import run
from settings import CANDIDATES, MANIFESTS

if __name__ == "__main__":
    run(["merge", "--root", CANDIDATES, "--output", MANIFESTS / "clips.csv"])
