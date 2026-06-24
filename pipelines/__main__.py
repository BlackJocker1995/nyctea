"""Pipeline entry-point dispatcher.

Each stage has a console entry point (``nyctea-<stage>``) defined in
``pyproject.toml`` that calls the matching ``<stage>_main`` here. All stages
share the same argument surface: ``--mode`` overrides ``NYCTEA_MODE`` /
``config.yaml``, ``--instances`` overrides ``NYCTEA_INSTANCES``, ``--debug``
toggles verbose logging.

The stages consume each other only via on-disk artefacts (npz shards for the
buffer, ``params{EXE}.csv`` from ICSearcher, ``params_repair{EXE}.csv`` for
repair results) — so each can be re-run independently, mirroring ICSearcher's
pipeline model.
"""
import argparse
import sys

from nyctea.config import toolConfig
from nyctea.logging_config import setup_logging


def _common_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--instances", type=int, default=None,
                   help="Number of concurrent simulator instances "
                        "(overrides NYCTEA_INSTANCES / config.yaml).")
    p.add_argument("--debug", action="store_true", default=None,
                   help="Enable DEBUG logging (overrides NYCTEA_DEBUG).")
    return p


def _apply_overrides(args):
    """Push --instances / --debug into the environment before stage code runs.

    Mode is resolved at config import time (NYCTEA_MODE), so it is not
    re-parsed here; the other two are read live by the relevant subsystems.
    """
    import os
    if getattr(args, "instances", None) is not None:
        os.environ["NYCTEA_INSTANCES"] = str(args.instances)
    if getattr(args, "debug", None):
        os.environ["NYCTEA_DEBUG"] = "1"


def collect_main(argv=None):
    """Stage 1: collect RL training data (fly bad configs, gather transitions)."""
    args = _common_parser("nyctea collect: fly ICSearcher's destabilizing configs "
                          "and gather DDPG transitions into npz buffer shards.") \
        .parse_args(argv)
    _apply_overrides(args)
    setup_logging(debug=args.debug)
    from pipelines import collect
    collect.run(args)


def train_main(argv=None):
    """Stage 2: train the DDPG actor/critic on the collected buffer."""
    args = _common_parser("nyctea train: train the DDPG repair agent.") \
        .parse_args(argv)
    _apply_overrides(args)
    setup_logging(debug=args.debug)
    from pipelines import train
    train.run(args)


def repair_main(argv=None):
    """Stage 3: repair-test bad configs with the trained agent."""
    parser = _common_parser("nyctea repair: repair-test destabilizing configs "
                            "with the trained DDPG agent.")
    parser.add_argument("--disturbance", choices=("none", "wind", "sensor", "random"),
                        default="none",
                        help="Disturbance mode for the repair test "
                             "(replaces the legacy _wind/_sensor/_random scripts).")
    args = parser.parse_args(argv)
    _apply_overrides(args)
    setup_logging(debug=args.debug)
    from pipelines import repair
    repair.run(args)


def validate_main(argv=None):
    """Stage 4: re-fly the repaired configs to confirm stability."""
    args = _common_parser("nyctea validate: re-fly repaired configs to verify.") \
        .parse_args(argv)
    _apply_overrides(args)
    setup_logging(debug=args.debug)
    from pipelines import validate
    validate.run(args)


def analyze_main(argv=None):
    """Stage 5: loss / detection / figure analysis."""
    args = _common_parser("nyctea analyze: loss / detection / deviation analysis.") \
        .parse_args(argv)
    _apply_overrides(args)
    setup_logging(debug=args.debug)
    from pipelines import analyze
    analyze.run(args)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m pipelines <stage> [args]\n"
              "stages: collect, train, repair, validate, analyze",
              file=sys.stderr)
        sys.exit(0 if sys.argv[1:] and sys.argv[1] in ("-h", "--help") else 2)
    stage = sys.argv.pop(1)
    dispatch = {
        "collect": collect_main, "train": train_main, "repair": repair_main,
        "validate": validate_main, "analyze": analyze_main,
    }
    if stage not in dispatch:
        print(f"unknown stage {stage!r}; expected one of {list(dispatch)}",
              file=sys.stderr)
        sys.exit(2)
    dispatch[stage]()
