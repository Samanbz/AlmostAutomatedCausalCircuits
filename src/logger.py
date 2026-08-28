import logging

import colorlog


handler = colorlog.StreamHandler()
# 2. Optimized Format String:
# %(.8s)c -> Level name limited to 8 chars
# %-5s    -> Level name padded to 5 chars (DEBUG is 5, INFO is 4)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s | %(name)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

handler.setFormatter(formatter)
logger = logging.getLogger("mcc")
logger.setLevel(logging.WARNING)
logger.addHandler(handler)
logging.getLogger("graphviz").setLevel(logging.WARNING)
