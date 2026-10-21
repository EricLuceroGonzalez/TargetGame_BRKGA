"""Punto de entrada: ``uv run python -m ga_target_game``."""

import logging

from ga_target_game.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging(log_filename="ga_target_game.log")
    logger.info("ga-target-game iniciado")


if __name__ == "__main__":
    main()
