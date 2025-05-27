"""
Main script to run the Handover Optimization Framework.

This script provides a unified command-line interface to:
- Train a PPO-based handover policy.
- Validate handover performance using a PPO policy.
- Benchmark using a standard 3GPP-compliant handover algorithm.

Usage:
    python main.py <script>

Arguments:
    script: One of 'train_ppo', 'validate_ppo', or 'validate_3gpp'.
"""

import argparse
import os
import sys
from scripts import train_ppo, validate_3gpp, validate_ppo

THIS_PATH = os.path.dirname(os.path.abspath(__file__))


def run() -> int:
    """Run the Handover Optimization Framework."""
    parser = argparse.ArgumentParser(
        description="Handover Optimization Framework\n\n"
        "Use this entry point to train and evaluate different handover strategies:\n"
        "  • 3GPP-standard handover\n"
        "  • PPO-based handover\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="\nAvailable commands:")

    # validate_3gpp
    subparsers.add_parser(
        "validate_3gpp",
        help="Validate using 3GPP-compliant handover",
        description="Runs the 3GPP-standard handover validation procedure.",
    )

    # validate_ppo
    subparsers.add_parser(
        "validate_ppo",
        help="Validate using a trained PPO policy",
        description="Runs validation using a pre-trained PPO handover policy.",
    )

    # train_ppo
    subparsers.add_parser(
        "train_ppo",
        help="Train a PPO policy for handover decisions",
        description="Trains a PPO policy to make optimal handover decisions.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if args.command == "validate_3gpp":
        return validate_3gpp.main(THIS_PATH)
    if args.command == "validate_ppo":
        return validate_ppo.main(THIS_PATH)
    if args.command == "train_ppo":
        return train_ppo.main(THIS_PATH)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(run())
