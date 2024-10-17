# Copyright © 2024 Apple Inc.

"""Device monitor module, to collect and report system metrics."""
import contextlib
import threading
from typing import List, Literal

from absl import logging
from tpu_info import device

from axlearn.common.config import Configurable, config_class, maybe_instantiate
from axlearn.common.monitoring.tpu_mon_client import MetricV2Name as MetricName
from axlearn.common.monitoring.tpu_mon_client import Usage
from axlearn.common.monitoring.tpu_mon_client import get_chip_metrics_v2 as get_chip_metrics
from axlearn.common.monitoring.tpu_mon_client import (
    validate_available_metrics_v2 as validate_available_metrics,
)


class DeviceMonitorClient(Configurable):
    """Base Client for fetching metrics from devices."""

    @config_class
    class Config(Configurable.Config):
        """Configures DeviceMonitorClient."""

        # TODO(kelvin-zou): Add support for GPU and Trainium.
        platform: Literal["tpu", "gpu", "trainium"] = "tpu"

    def __init__(self, cfg: Config):
        """Initialize the DeviceMonitorClient."""
        super().__init__(cfg)
        cfg = self.config
        self._platform = cfg.platform

    def collect_metrics(self) -> List[Usage]:
        """Collect metrics from the device, it should be empty."""
        return []

    def is_host_idle(self, usages: List[Usage]) -> bool:
        """Check if the devices on the host are idle, always return False."""
        # Make sure the usages are empty.
        assert usages == []
        return False


class TpuMonitorClient(DeviceMonitorClient):
    """Client for fetching TPU metrics from libtpu."""

    @config_class
    class Config(DeviceMonitorClient.Config):
        """Configures TpuMonitorClient."""

        # At the moment all architecures have 4 chips per node, so chip_type doesn't matter here.
        chip_type: device.TpuChip = device.TpuChip.V5P
        # We log the two metrics currently,
        # since we use them to determine the idle status of the host.
        metric_list: List[MetricName] = [
            MetricName.MEMORY_BANDWIDTH_UTILIZATION,
            MetricName.TENSORCORE_UTIL_PCT,
        ]
        addr: str = "localhost:2112"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._metric_list = cfg.metric_list
        self._addr = cfg.addr
        self._chip_type = cfg.chip_type
        self._enabled = validate_available_metrics(self._metric_list, self._addr)

    def collect_metrics(self) -> List[Usage]:
        """Collect TPU metrics."""
        if not self._enabled:  # Return empty list if we see any unsupported metrics.
            return []
        usages = get_chip_metrics(self._chip_type, self._metric_list, self._addr)
        # TODO(kelvin-zou): get DCN metrics from container.
        return usages

    def is_host_idle(self, usages: List[Usage]) -> bool:
        """Check if the TPU device on the host are idle."""
        for usage in usages:
            if usage.mem_bw_util_pct <= 0.1 and usage.tensorcore_util_pct <= 0.1:
                logging.info("TPU device %d is idle.", usage.device_id)
                return True
        return False


class DeviceMonitor(Configurable):
    """Device Monitor to collect and report system metrics.
    It also checks if the devices on the host are idle.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures DeviceMonitor."""

        # The interval to report the system metrics, in seconds. 0 to disable.
        monitor_client_cfg: DeviceMonitorClient.Config = DeviceMonitorClient.default_config()
        report_interval_in_seconds: float = 60  # default querying every 60 seconds.
        log_every_n: int = 20  # default logging every 20 queries, i.e. every 20 minutes.

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._monitor_client = maybe_instantiate(cfg.monitor_client_cfg)
        self._platform = cfg.monitor_client_cfg.platform
        self._log_every_n = cfg.log_every_n
        self._report_interval = cfg.report_interval_in_seconds
        self._idle = False
        self._monitor_thread = None
        self._monitor_stopping = None

    @contextlib.contextmanager
    def start_monitoring(self):
        """Start the monitor."""
        self._start_monitoring()
        try:
            yield
        finally:
            self._stop_monitor()

    def is_host_idle(self) -> bool:
        """Check if the TPU device on the host are idle."""
        return self._idle

    def _check_host_and_log_metrics(self) -> bool:
        """Check if the devices on the host are idle."""
        metrics: List[Usage] = self._monitor_client.collect_metrics()
        logging.log_every_n(
            logging.INFO, "%s metrics: %s", self._log_every_n, self._platform, metrics
        )
        return self._monitor_client.is_host_idle(metrics)

    def _start_monitoring(self):
        """Start the monitor."""
        if self._report_interval > 0:
            self._monitor_stopping = threading.Event()
            self._monitor_thread = threading.Thread(
                name="tpu_device_monitor",
                target=self._monitor_loop,
            )
            self._monitor_thread.start()
            logging.info("_monitor_thread started.")

    def _stop_monitor(self):
        """Stops the monitor."""
        logging.info("Waiting for watchdog_thread to finish")
        if self._monitor_thread is not None:
            self._monitor_stopping.set()
            self._monitor_thread.join()
            self._monitor_thread = None
            logging.info("_monitor_thread finished.")

    def _monitor_loop(self):
        cfg = self.config
        while True:
            # Update the idle status.
            self._idle = self._check_host_and_log_metrics()
            if self._monitor_stopping.wait(timeout=cfg.report_interval_in_seconds):
                break
        logging.info("mointor loop exit.")
