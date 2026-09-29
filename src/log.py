"""Logging set-up shared by the command-line entry points."""

import logging

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Send timestamped progress messages from all modules to stderr."""
    logging.basicConfig(level=level, format=FORMAT, datefmt="%H:%M:%S")
