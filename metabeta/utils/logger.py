import logging
import sys


def setupLogging(verbosity: int) -> None:
    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }

    log_level = level_map.get(verbosity, logging.DEBUG)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
