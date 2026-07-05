from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-backend", default="auto")
    args = parser.parse_args()
    Path("torch_backend.txt").write_text(str(args.torch_backend), encoding="utf-8")
    print(f"[patch_env] torch backend set to {args.torch_backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
