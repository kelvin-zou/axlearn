# Copyright © 2024 Apple Inc.

# Some of the code in this file is adapted from:
# https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/main/tpu_info/tpu_info/metrics_test.py

"""Test for tpu_client."""
import contextlib
import functools
import http.server
import os
import threading
from concurrent import futures
from typing import Dict, List, Union

import grpc
from absl import logging
from absl.testing import parameterized
from tpu_info import device
from tpu_info.proto import tpu_metric_service_pb2 as tpu_metrics
from tpu_info.proto import tpu_metric_service_pb2_grpc as tpu_metrics_grpc

from axlearn.common.monitoring import tpu_mon_client

V5P_METRICS = [
    "tpu.runtime.hbm.memory.usage.bytes",
    "tpu.runtime.hbm.memory.total.bytes",
    "tpu.runtime.tensorcore.dutycycle.percent",
    "tpu.runtime.uptime.seconds.gauge",
    "megascale.dcn_transfer_latencies.microsecond.cumulative.distribution",
    "megascale.collective_end_to_end_latencies.microsecond.cumulative.distribution",
]


testdata_dir: str = os.path.join(os.path.dirname(__file__), "./testdata")


class FakeTpuGrpcService(tpu_metrics_grpc.RuntimeMetricServiceServicer):
    """Fake libtpu server for testing."""

    def __init__(
        self,
        responses: Dict[tpu_mon_client.MetricName, List[Union[int, float]]],
        available_metrics: List[str],
    ):
        """initialize the fake libtpu server.

        Args:
            responses (Dict[tpu_client.MetricName, List[Union[int, float]]]):
                    The expected responses for the metrics.
        """
        self._responses = responses
        self._available_metrics = available_metrics
        super().__init__()

    @functools.singledispatchmethod
    def _gauge(self, val):
        """dummy function to create Gauge from different types.

        Args:
            val: The value to create Gauge from.

        Raises:
            NotImplementedError: Cannot create Gauge from the given type.
        """
        raise NotImplementedError(f"Cannot create Gauge from {val}")

    @_gauge.register
    def _(self, val: int):
        """Create Gauge from int."""
        return tpu_metrics.Gauge(as_int=val)

    @_gauge.register
    def _(self, val: float):
        """Create Gauge from float."""
        return tpu_metrics.Gauge(as_double=val)

    def GetRuntimeMetric(self, request: tpu_metrics.MetricRequest, context):
        """Get the metric from the fake libtpu server."""
        metric_name = tpu_mon_client.MetricName(request.metric_name)
        resp = self._responses[metric_name]

        return tpu_metrics.MetricResponse(
            metric=tpu_metrics.TPUMetric(
                name=metric_name.value,
                metrics=[
                    tpu_metrics.Metric(
                        attribute=tpu_metrics.Attribute(
                            key="device-id", value=tpu_metrics.AttrValue(int_attr=i)
                        ),
                        gauge=self._gauge(val),
                    )
                    for i, val in enumerate(resp)
                ],
            )
        )

    def ListSupportedMetrics(self, request: tpu_metrics.ListSupportedMetricsRequest, context):
        """List the supported metrics from the fake libtpu server."""
        # The test supported metrics are based on V5P libtpu.
        supported_metrics = [
            tpu_metrics.SupportedMetric(metric_name=metric) for metric in self._available_metrics
        ]
        return tpu_metrics.ListSupportedMetricsResponse(supported_metric=supported_metrics)


class DummyTpuMetricServer:
    """Dummy TPU metric server for testing."""

    @staticmethod
    @contextlib.contextmanager
    def fake_tpu_metrics(responses, metric_list):
        """Create a fake libtpu server for testing."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        tpu_metrics_grpc.add_RuntimeMetricServiceServicer_to_server(
            FakeTpuGrpcService(responses, metric_list), server
        )
        port = server.add_secure_port("localhost:0", grpc.local_server_credentials())
        try:
            server.start()
            yield f"localhost:{port}"
        finally:
            server.stop(None)


class TestMetrics(parameterized.TestCase):
    """Test class for TPU client methods."""

    def test_validation(self):
        """Test the validation of the available metrics."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        tpu_metrics_grpc.add_RuntimeMetricServiceServicer_to_server(
            FakeTpuGrpcService({}, available_metrics=V5P_METRICS), server
        )
        port = server.add_secure_port("localhost:0", grpc.local_server_credentials())

        fake_libtpu_addr = f"localhost:{port}"
        server.start()
        validate = tpu_mon_client.validate_available_metrics(
            list(tpu_mon_client.MetricName), fake_libtpu_addr
        )
        self.assertTrue(validate)
        server.stop(None)

    def test_validation_false(self):
        """Test the validation of the available metrics."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        tpu_metrics_grpc.add_RuntimeMetricServiceServicer_to_server(
            FakeTpuGrpcService({}, available_metrics=["fake_metrics"]), server
        )
        port = server.add_secure_port("localhost:0", grpc.local_server_credentials())
        fake_libtpu_addr = f"localhost:{port}"
        server.start()
        validate = tpu_mon_client.validate_available_metrics(
            list(tpu_mon_client.MetricName), fake_libtpu_addr
        )
        self.assertFalse(validate)
        server.stop(None)

    @parameterized.named_parameters(
        [
            (
                "v4_8",
                device.TpuChip.V4,
                {
                    tpu_mon_client.MetricName.TOTAL_MEMORY: [34088157184] * 4,
                    tpu_mon_client.MetricName.MEMORY_USAGE: [8522039296 * i for i in range(4)],
                    tpu_mon_client.MetricName.DUTY_CYCLE_PCT: [0.0, 25.0, 50.0, 100.0],
                },
            ),
            (
                "v5litepod_4",
                device.TpuChip.V5E,
                {
                    tpu_mon_client.MetricName.TOTAL_MEMORY: [17044078592] * 4,
                    tpu_mon_client.MetricName.MEMORY_USAGE: [2130509824 * i for i in range(4)],
                    tpu_mon_client.MetricName.DUTY_CYCLE_PCT: [100.0 / (i + 1) for i in range(4)],
                },
            ),
            (
                "v5p_8",
                device.TpuChip.V5P,
                {
                    tpu_mon_client.MetricName.TOTAL_MEMORY: [102803439616] * 4,
                    tpu_mon_client.MetricName.MEMORY_USAGE: [17044078592 * i for i in range(4)],
                    # Duty cycle is reported per-chip
                    tpu_mon_client.MetricName.DUTY_CYCLE_PCT: [100.0 / (i + 1) for i in range(4)],
                },
            ),
        ],
    )
    def test_metrics(self, chip_type: device.TpuChip, responses):
        """Test the metrics from the fake libtpu server."""
        with DummyTpuMetricServer.fake_tpu_metrics(responses, V5P_METRICS) as fake_libtpu_addr:
            expected_usage = [
                tpu_mon_client.Usage(i, m, t, d)
                for i, (m, t, d) in enumerate(
                    zip(
                        responses[tpu_mon_client.MetricName.MEMORY_USAGE],
                        responses[tpu_mon_client.MetricName.TOTAL_MEMORY],
                        responses[tpu_mon_client.MetricName.DUTY_CYCLE_PCT],
                    )
                )
            ]
            # Get the metrics from the fake libtpu server.
            chip_metrics = tpu_mon_client.get_chip_metrics(
                chip_type, list(tpu_mon_client.MetricName), fake_libtpu_addr
            )
            self.assertListEqual(chip_metrics, expected_usage)


class CustomHTTPServer(http.server.HTTPServer):
    metric_file: str

    def __init__(self, server_address, RequestHandlerClass, metric_file: str):
        super().__init__(server_address, RequestHandlerClass)
        self.metric_file = metric_file  # Store the custom parameter


class FakeTpuDevicePluginHandler(http.server.SimpleHTTPRequestHandler):
    """Fake TPU device plugin server for testing."""

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("charset", "utf-8")
            self.end_headers()
            if hasattr(self.server, "metric_file"):
                # pylint: disable-next=protected-access
                file_path = os.path.join(testdata_dir, self.server.metric_file)
                # pylint: enable-next=protected-access
            else:
                raise ValueError("No metric file provided.")
            with open(file_path, "rb") as f:
                self.wfile.write(f.read().decode("utf-8").encode("utf-8"))


class DummyTpuMetricV2Server:
    """Dummy TPU metric server for testing."""

    @staticmethod
    def fake_tpu_metrics_v2(metric_file: str = "sample_metrics.txt"):
        """Create a fake libtpu server for testing."""

        server_started = threading.Event()
        port = 0

        def start_dummy_server():
            with CustomHTTPServer(
                ("", 0), FakeTpuDevicePluginHandler, metric_file=metric_file
            ) as httpd:
                nonlocal port
                port = httpd.server_address[1]
                server_started.set()
                try:
                    httpd.serve_forever()
                finally:
                    httpd.shutdown()

        server_thread = threading.Thread(target=start_dummy_server)
        server_thread.daemon = True  # Makes the thread exit when the main program exits
        server_thread.start()

        server_started.wait()
        return f"localhost:{port}"


class TestMetricsV2(parameterized.TestCase):
    """Test class for TPU metrics v2 methods."""

    def test_all(self):
        """Test the validation of the available metrics."""
        fake_metrics_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2()
        logging.info("Fake metrics server started at %s", fake_metrics_addr)

        # Good metric list.
        validate = tpu_mon_client.validate_available_metrics_v2(
            list(tpu_mon_client.MetricV2Name), fake_metrics_addr
        )
        self.assertTrue(validate)

        # Bad metric list.
        validate = tpu_mon_client.validate_available_metrics_v2(
            list(tpu_mon_client.MetricName), fake_metrics_addr
        )
        self.assertFalse(validate)

        result = tpu_mon_client.get_chip_metrics_v2(
            device.TpuChip.V5P, list(tpu_mon_client.MetricV2Name), fake_metrics_addr
        )
        self.assertEqual(len(result), 4)
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
        self.assertListEqual(result, expected_usage)

        # Query a bad metric server.
        result = tpu_mon_client.get_chip_metrics_v2(
            device.TpuChip.V5P, list(tpu_mon_client.MetricName), "localhost:8080"
        )
        self.assertListEqual(result, [])
