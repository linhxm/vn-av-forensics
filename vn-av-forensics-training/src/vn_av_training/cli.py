"""Public entry point for the supported audio-visual relation workflow."""

import sys


def main(argv=None):
    from vn_av_training.relation_cli import main as dispatch

    return dispatch(sys.argv[1:] if argv is None else argv)


def entrypoint():
    try:
        return main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
