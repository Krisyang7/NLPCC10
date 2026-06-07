#!/usr/bin/env python3
"""Run the Track 1 / Track 2 prompting baseline against an Anthropic endpoint.

Same CLI as baseline_prompting/run_baseline.py, but it injects the
AnthropicCompatibleClient so requests go to the Anthropic Messages API
(e.g. the aicoding relay). Credentials default to the same env vars Claude
Code already uses:

    ANTHROPIC_BASE_URL   (default: https://api.aicoding.sh)
    ANTHROPIC_AUTH_TOKEN

Example:
    python run_aicoding.py --track 1 \
        --input outputs/track1_dev_input.jsonl \
        --output outputs/track1_dev_pred.jsonl \
        --model claude-sonnet-4-6 --workers 4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_prompting.anthropic_client import AnthropicCompatibleClient
from baseline_prompting.errors import BaselineError
from baseline_prompting.run_baseline import build_arg_parser
from baseline_prompting.runner import run_baseline


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.aicoding.sh")
    api_key = args.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    if not args.model or args.model == "gpt-4o-mini":
        args.model = os.environ.get("AICODING_MODEL", "claude-sonnet-4-6")

    try:
        client = AnthropicCompatibleClient(base_url, api_key, timeout=args.timeout)
        return run_baseline(args, client=client)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
