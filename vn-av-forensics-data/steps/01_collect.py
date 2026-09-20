from _run import run
from settings import SOURCES

if __name__ == "__main__":
    run(["collect", "--input", SOURCES / "videos.csv", "--output", SOURCES / "selected_videos.csv"])
