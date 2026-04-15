import logging
from typing import Any

__all__ = ["logger"]

logger = logging.getLogger("map_pfn")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)


def device_info_filter(record: Any) -> bool:
    return "PU available: " not in record.getMessage()


logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(device_info_filter)
