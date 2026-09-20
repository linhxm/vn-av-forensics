from _run import run
from settings import CANDIDATES, MANIFESTS

if __name__ == "__main__":
    run(["review", "--root", CANDIDATES, "--review", MANIFESTS / "clips.csv"])
