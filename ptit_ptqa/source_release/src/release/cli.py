from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_release_config
from .replay import build_replay_cache, replay_submission, verify_release


def main() -> None:
    parser = argparse.ArgumentParser(description="R2AI Stage 3 audited release CLI")
    parser.add_argument("--config", type=Path, default=Path("configs/release_exp143.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Verify release inputs and prediction fingerprint")
    verify.add_argument("--fast", action="store_true", help="Skip streaming the full prediction payload")
    cache = commands.add_parser("build-cache", help="Build the compressed UI replay cache")
    cache.add_argument("--force", action="store_true")
    replay = commands.add_parser("replay", help="Copy the audited champion to a new immutable output")
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_release_config(args.config)
    if args.command == "verify":
        result = verify_release(config, verify_predictions=not args.fast)
    elif args.command == "build-cache":
        result = build_replay_cache(config, force=args.force)
    else:
        result = replay_submission(config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
