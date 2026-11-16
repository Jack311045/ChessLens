"""Run the local single-worker API with an explicit bundle."""

import argparse
from pathlib import Path

import uvicorn

from chesslens.serving.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Defaults to CHESSLENS_BUNDLE_DIR")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=2, help="ORT intra-op CPU threads")
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.bundle, threads=args.threads),
        host="127.0.0.1",
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
