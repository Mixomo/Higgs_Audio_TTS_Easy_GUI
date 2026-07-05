from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated Higgs ASR worker.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="Auto-detect")
    args = parser.parse_args()

    os.environ["HIGGS_ASR_WORKER"] = "1"
    os.environ["HIGGS_NO_COMPILE"] = "1"
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from higgs_local.asr import ASRManager

        text, status = ASRManager().transcribe(args.audio, args.model, args.language)
        print(json.dumps({"ok": True, "text": text, "status": status}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=4)},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
