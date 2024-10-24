# Copyright © 2024 Apple Inc.
""" TPU Device Monitor for fetching TPU metrics from libtpu and tpu-device-plugin. """

from typing import Sequence

from absl import logging
from tpu_info import device

from axlearn.cloud.gcp.monitoring.tpu_client import MetricV2Name as MetricName
from axlearn.cloud.gcp.monitoring.tpu_client import get_chip_metrics_v2 as get_chip_metrics
from axlearn.cloud.gcp.monitoring.tpu_client import (
    validate_available_metrics_v2 as validate_available_metrics,
)
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.monitoring.device_monitor import DeviceMonitorClient
from axlearn.common.utils import DeviceUsage as Usage

DEVICE_KIND_TO_CHIP_TYPE = {
    "TPU v4": device.TpuChip.V4,
    "TPU v5": device.TpuChip.V5P,
    "TPU v5e": device.TpuChip.V5E,
    "TPU v6e": device.TpuChip.V6E,
}


class TPUMonitorClient(DeviceMonitorClient):
    """Client for fetching TPU metrics from libtpu."""

    @config_class
    class Config(DeviceMonitorClient.Config):
        """Configures TPUMonitorClient."""

        # At the moment all architecures have 4 chips per node,
        # so chip_type doesn't matter much here.
        chip_type: Required[device.TpuChip] = REQUIRED
        # We log the two metrics currently,
        # since we use them to determine the idle status of the host.
        metric_list: Sequence[MetricName] = [
            MetricName.MEMORY_BANDWIDTH_UTILIZATION,
            MetricName.TENSORCORE_UTIL_PCT,
        ]
        addr: str = "localhost:2112"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._enabled = validate_available_metrics(cfg.metric_list, addr=cfg.addr)
        logging.log_if(logging.ERROR, not self._enabled, "TPU metrics are not supported.")

    def collect_metrics(self) -> list[Usage]:
        """Collect TPU metrics."""
        if not self._enabled:  # Return empty list if we see any unsupported metrics.
            return []
        cfg: TPUMonitorClient.Config = self.config
        usages = get_chip_metrics(cfg.metric_list, chip_type=cfg.chip_type, addr=cfg.addr)
        # TODO(kelvin-zou): get DCN metrics from container.
        return usages

    def is_host_idle(self, usages: list[Usage]) -> bool:
        """Check if the TPU device on the host are idle."""
        for usage in usages:
            if usage.mem_bw_util_pct <= 0.1 and usage.tensorcore_util_pct <= 0.1:
                logging.info("TPU device %d is idle.", usage.device_id)
                return True
        return False
