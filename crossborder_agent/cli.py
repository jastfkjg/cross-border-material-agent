"""Command-line interface required by the evaluation harness."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import VERSION
from .availability import create_last_resort_delivery
from .input_loader import InputError, parse_prompt_paths
from .logging_utils import configure_logging
from .pipeline import Pipeline, PipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-border product localization material agent"
    )
    parser.add_argument(
        "--version", action="store_true", help="print semantic version and exit"
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="evaluation task prompt containing input/output paths",
    )
    parser.add_argument(
        "--input-dir", default="", help="development override for input directory"
    )
    parser.add_argument(
        "--output-dir", default="", help="development override for output directory"
    )
    parser.add_argument(
        "--product-id",
        default="",
        help="select one product when a development dataset has several",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip model calls and exercise deterministic fallbacks (development only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="write structured, redacted TRACE_JSON events to agent_debug.jsonl (local diagnostics)",
    )
    parser.add_argument(
        "--run-profile",
        choices=("full", "fast"),
        default=os.environ.get("AGENT_RUN_PROFILE", "full").strip().lower() or "full",
        help="full keeps submission behavior; fast reduces model calls for local iteration",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=29 * 60,
        help="internal deadline; defaults to one minute below the platform limit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(VERSION)
        return 0

    debug = args.debug or os.environ.get("AGENT_DEBUG", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    logger = configure_logging(debug=debug)
    input_dir: Path | None = None
    output_dir: Path | None = None
    try:
        if args.input_dir or args.output_dir:
            if not args.input_dir or not args.output_dir:
                raise InputError("--input-dir and --output-dir must be provided together")
            input_dir = Path(args.input_dir).expanduser().resolve()
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            if not args.prompt:
                raise InputError(
                    "Provide --prompt, or provide both development input/output directories"
                )
            paths = parse_prompt_paths(args.prompt)
            input_dir, output_dir = paths.input_dir, paths.output_dir

        if args.timeout_seconds < 60:
            raise InputError("--timeout-seconds cannot be less than 60")
        logger.info(
            "Starting Agent: input=%s output=%s offline=%s profile=%s",
            input_dir,
            output_dir,
            args.offline,
            args.run_profile,
        )
        pipeline = Pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            product_id=args.product_id.strip(),
            logger=logger,
            timeout_seconds=args.timeout_seconds,
            offline=args.offline,
            debug=debug,
            run_profile=args.run_profile,
        )
        pipeline.run()
        return 0
    except KeyboardInterrupt:
        logger.error("Agent was interrupted")
        return 130
    except Exception as exc:  # final defensive boundary for the evaluator
        logger.exception("Agent execution failed: %s", exc)
        if output_dir is not None and create_last_resort_delivery(
            output_dir,
            logger=logger,
            reason=f"{type(exc).__name__}: {exc}",
            input_dir=input_dir,
            product_id=args.product_id.strip(),
        ):
            return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())
