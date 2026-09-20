import os
import sys

from settings import ROOT

sys.path.insert(0, str(ROOT / "src"))
from vn_av_data.cli import main


def run(args):
    os.chdir(ROOT)
    raise SystemExit(main([str(x) for x in args] + sys.argv[1:]))
