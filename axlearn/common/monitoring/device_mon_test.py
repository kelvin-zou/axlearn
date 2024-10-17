# Copyright © 2024 Apple Inc.

"""Test class for device_mon.py."""
import time

from absl.testing import parameterized
from tpu_info import device

from axlearn.common.monitoring import tpu_mon_client
from axlearn.common.monitoring.device_mon import DeviceMonitor, TpuMonitorClient
from axlearn.common.monitoring.tpu_mon_client_test import DummyTpuMetricV2Server


class TestMetrics(parameterized.TestCase):
    """Test class for TpuMonitorClient and DeviceMonitor."""

    def test_tpu_client(self):
        """Test the TpuMonitorClient."""
        expected_usage = [
            tpu_mon_client.Usage(
                device_id=i,
                mem_total_in_bytes=int(1.02803439616e11),
                mem_used_in_bytes=int(6.5e10),
                duty_cycle_pct=100.0,
                tensorcore_util_pct=1.0 * (1 + i),
                mem_bw_util_pct=30.0,
            )
            for i in range(4)
        ]
        # Test the case where the metrics are supported.
        metric_server_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2()
        tpu_monitor_client_cfg = TpuMonitorClient.default_config().set(
            metric_list=list(tpu_mon_client.MetricV2Name),
            chip_type=device.TpuChip.V5P,
            addr=metric_server_addr,
        )
        tpu_monitor_client = tpu_monitor_client_cfg.instantiate()
        chip_metrics = tpu_monitor_client.collect_metrics()
        self.assertListEqual(chip_metrics, expected_usage)
        self.assertFalse(tpu_monitor_client.is_host_idle(chip_metrics))

    def test_tpu_client_no_metric_supported(self):
        """Test the TpuMonitorClient when no metric is supported."""
        metric_server_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2()
        tpu_monitor_client_cfg = TpuMonitorClient.default_config().set(
            metric_list=list(tpu_mon_client.MetricName),
            chip_type=device.TpuChip.V5P,
            addr=metric_server_addr,
        )
        tpu_monitor_client = tpu_monitor_client_cfg.instantiate()
        chip_metrics = tpu_monitor_client.collect_metrics()
        self.assertListEqual(chip_metrics, [])
        self.assertFalse(tpu_monitor_client.is_host_idle(chip_metrics))

    def test_device_monitor(self):
        """Test the TpuMonitorClient."""
        metric_server_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2()
        tpu_monitor_client_cfg = TpuMonitorClient.default_config().set(
            chip_type=device.TpuChip.V5P,
            addr=metric_server_addr,
        )
        device_monitor_cfg = DeviceMonitor.default_config().set(
            monitor_client_cfg=tpu_monitor_client_cfg,
            report_interval_in_seconds=0.1,
            log_every_n=1,
        )
        device_monitor = device_monitor_cfg.instantiate()
        with device_monitor.start_monitoring():
            time.sleep(0.2)
            self.assertFalse(device_monitor.is_host_idle())

    def test_device_monitor_idle(self):
        """Test the TpuMonitorClient."""
        metric_server_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2(
            metric_file="sample_metrics_idle.txt"
        )
        tpu_monitor_client_cfg = TpuMonitorClient.default_config().set(
            chip_type=device.TpuChip.V5P,
            addr=metric_server_addr,
        )
        device_monitor_cfg = DeviceMonitor.default_config().set(
            monitor_client_cfg=tpu_monitor_client_cfg,
            report_interval_in_seconds=0.1,
            log_every_n=1,
        )
        device_monitor = device_monitor_cfg.instantiate()
        with device_monitor.start_monitoring():
            time.sleep(0.2)
            self.assertTrue(device_monitor.is_host_idle())
