"""Main script to run the Handover Optimization Framework."""

import argparse
import os
import sys

from scripts.validate_3gpp import main as validate_3gpp
from scripts.validate_ppo import main as validate_ppo


THIS_PATH = os.path.dirname(os.path.abspath(__file__))


def run():
    """Run the Handover Optimization Framework."""
    parser = argparse.ArgumentParser(description="Handover Optimization Framework")
    parser.add_argument(
        "script",
        choices=["validate_3gpp", "validate_ppo"],
        help="Select the script to run",
    )
    args = parser.parse_args()

    if args.script == "validate_3gpp":
        validate_3gpp(THIS_PATH)
    elif args.script == "validate_ppo":
        validate_ppo(THIS_PATH)


if __name__ == "__main__":
    sys.exit(run())
