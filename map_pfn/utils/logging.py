import logging
from typing import Any

__all__ = ["logger"]

logger = logging.getLogger()


def device_info_filter(record: Any) -> bool:
    return "PU available: " not in record.getMessage()


logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(device_info_filter)
