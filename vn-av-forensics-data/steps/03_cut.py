from _run import run
from settings import CUT, RAW

if __name__ == "__main__":
    run(["cut", "--manifest", RAW / "sources.jsonl", "--output", CUT])
