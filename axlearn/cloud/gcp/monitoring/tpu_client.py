# Copyright © 2024 Apple Inc.

# Some of the code in this file is adapted from:
# AI-Hypercomputer/cloud-accelerator-diagnostics:
# Copyright 2023 Google LLC
# https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/main/tpu_info/tpu_info/metrics.py

"""Client for fetching TPU metrics from libtpu."""
import enum
import urllib.request
from typing import Sequence

import grpc
from absl import logging
from prometheus_client.core import Metric
from prometheus_client.parser import text_string_to_metric_families
from tpu_info import device
from tpu_info.proto import tpu_metric_service_pb2 as tpu_metrics
from tpu_info.proto import tpu_metric_service_pb2_grpc as tpu_metrics_grpc

from axlearn.common.utils import DeviceUsage as Usage


# Interface names for libtpu metrics.
class MetricName(enum.Enum):
    """Metric names defined in libtpu."""

    DUTY_CYCLE_PCT = "tpu.runtime.tensorcore.dutycycle.percent"
    TOTAL_MEMORY = "tpu.runtime.hbm.memory.total.bytes"
    MEMORY_USAGE = "tpu.runtime.hbm.memory.usage.bytes"


# Interface names for tpu-device-plugin, which are plumbed from sources like libtpu.
class MetricV2Name(enum.Enum):
    """Metric names defined in tpu-device-plugin."""

    TOTAL_MEMORY = "memory_total"
    MEMORY_USAGE = "memory_used"
    DUTY_CYCLE_PCT = "duty_cycle_node"
    TENSORCORE_UTIL_PCT = "tensorcore_utilization"
    MEMORY_BANDWIDTH_UTILIZATION = "memory_bandwidth_utilization"


# At the moment all architecures have 4 chips per node.
# We may need to revisit this assumption for future gens.
CHIPS_PER_NODE = {
    device.TpuChip.V4: 4,
    device.TpuChip.V5E: 4,
    device.TpuChip.V5P: 4,
    device.TpuChip.V6E: 4,
}


# TODO(Kelvin-zou): will revisit this function to make it more libtpu native.
def validate_available_metrics(
    metric_list: Sequence[MetricName], *, addr: str = "localhost:8431"
) -> bool:
    """Validate the available metrics against the supported metrics from libtpu.

    Args:
        metric_list: the metrics to be fetched from the libtpu.
        addr: grpc server from libtpu. Defaults to "localhost:8431".

    Returns:
        bool: True if all metrics are supported, False otherwise.
    """
    # Considering the low cost of opening a new grpc client for each call,
    # we don't cache the client.
    channel = grpc.secure_channel(addr, grpc.local_channel_credentials())
    client = tpu_metrics_grpc.RuntimeMetricServiceStub(channel)

    # Manually annotate type until GRPC supports annotations
    # See https://github.com/grpc/grpc/issues/29041
    resp: tpu_metrics.MetricResponse = client.ListSupportedMetrics(
        tpu_metrics.ListSupportedMetricsRequest()
    )
    # Log all supported metrics, this should be only called once.
    logging.info("Supported metrics: %s", resp.supported_metric)
    supported_metric_names = [
        supported_metric.metric_name for supported_metric in resp.supported_metric
    ]
    # Validate metric_list against the supported metrics.
    is_valid = True
    for metric_name in metric_list:
        if metric_name.value not in supported_metric_names:
            logging.error("Metric %s is not supported.", metric_name.value)
            is_valid = False
    if is_valid:
        logging.info("Supported metrics: %s", supported_metric_names)
    return is_valid


# TODO(Kelvin-zou): will revisit this function to make it more libtpu native.
def get_chip_metrics(
    metric_list: Sequence[MetricName], *, chip_type: device.TpuChip, addr: str = "localhost:8431"
) -> list[Usage]:
    """Gets usage statistics for all attached TPU devices from libtpu.

    Args:
        metric_list: List of metrics to fetch from libtpu.
        chip_type: TPU chip version. Determines how metrics are interpreted.
        addr: GRPC server address of libtpu metrics server.

    Returns:
        List of usage statistics for each TPU device.
    """
    # Considering the low cost of opening a new grpc client for each call, we do live query.
    # The tcp connection may be cached by a lower level
    channel = grpc.secure_channel(addr, grpc.local_channel_credentials())
    client = tpu_metrics_grpc.RuntimeMetricServiceStub(channel)

    def sorted_metric_response(
        metric_name: str,
    ) -> list[tpu_metrics.Metric]:
        # Manually annotate type until GRPC supports annotations
        # See https://github.com/grpc/grpc/issues/29041
        resp: tpu_metrics.MetricResponse = client.GetRuntimeMetric(
            tpu_metrics.MetricRequest(metric_name=metric_name)
        )
        return sorted(resp.metric.metrics, key=lambda m: m.attribute.value.int_attr)

    metric_results = [Usage(device_id=i) for i in range(CHIPS_PER_NODE[chip_type])]
    for metric_name in metric_list:
        metric_result = sorted_metric_response(metric_name.value)

        if CHIPS_PER_NODE[chip_type] != len(metric_result):
            raise SystemError("Metrics not found for all chips, this indicates a serious issue.")

        if metric_name == MetricName.TOTAL_MEMORY:
            for i, metric in enumerate(metric_result):
                metric_results[i].mem_total_in_bytes = metric.gauge.as_int
        elif metric_name == MetricName.MEMORY_USAGE:
            for i, metric in enumerate(metric_result):
                metric_results[i].mem_used_in_bytes = metric.gauge.as_int
        elif metric_name == MetricName.DUTY_CYCLE_PCT:
            for i, metric in enumerate(metric_result):
                metric_results[i].duty_cycle_pct = metric.gauge.as_double

    return metric_results


def validate_available_metrics_v2(
    metric_list: Sequence[MetricName], *, addr: str = "localhost:2112"
) -> bool:
    """Validate the available metrics against the supported metrics from tpu device plugin.

    Args:
        metric_list: the metrics to be fetched from the tpu device plugin.
        addr: Address of tpu-device-plugin metrics server. Defaults to "localhost:2112".

    Returns:
        True if all metrics are supported, False otherwise.
    """
    # Due to no official way to list all metrics,
    # we do a live query and check if the metrics are supported.
    try:
        with urllib.request.urlopen(f"http://{addr}/metrics") as response:
            contents = response.read().decode("utf-8")
            families = list(text_string_to_metric_families(contents))
            supported_metrics = set()
            for family in families:
                if isinstance(family, Metric):
                    supported_metrics.add(family.name)
            is_valid = True
            for metric in metric_list:
                if metric.value not in supported_metrics:
                    logging.error("Metric %s is not supported.", metric.value)
                    is_valid = False
            if is_valid:
                logging.info("Supported metrics: %s", supported_metrics)
            return is_valid
    except urllib.error.URLError as e:
        logging.log_first_n(logging.ERROR, "Failed to fetch metrics from %s: %s", 5, addr, e)
        return False


def get_chip_metrics_v2(
    metric_list: Sequence[MetricV2Name], *, chip_type: device.TpuChip, addr: str = "localhost:2112"
) -> list[Usage]:
    """Gets usage statistics for tpu devices on the node, from tpu-device-plugin.

    Args:
        metric_list: List of metrics to fetch from tpu-device-plugin.
        chip_type: TPU chip version. Determines how metrics are interpreted.
        addr: Address of tpu-device-plugin metrics server, default at 2112 port.

    Returns:
        List of usage statistics for each TPU device
    """
    devices_per_node = CHIPS_PER_NODE[chip_type]
    # Consider the low cost of opening a new connection for each call, we do live query.
    try:
        with urllib.request.urlopen(f"http://{addr}/metrics") as response:
            contents = response.read().decode("utf-8")
            families = list(text_string_to_metric_families(contents))
            metric_results = [Usage(device_id=i) for i in range(CHIPS_PER_NODE[chip_type])]
            for family in families:
                if isinstance(family, Metric) and family.name in [i.value for i in metric_list]:
                    assert len(family.samples) == devices_per_node
                    for i, metric in enumerate(family.samples):
                        if family.name == MetricV2Name.TOTAL_MEMORY.value:
                            metric_results[i].mem_total_in_bytes = metric[2]
                        elif family.name == MetricV2Name.MEMORY_USAGE.value:
                            metric_results[i].mem_used_in_bytes = metric[2]
                        elif family.name == MetricV2Name.DUTY_CYCLE_PCT.value:
                            metric_results[i].duty_cycle_pct = metric[2]
                        elif family.name == MetricV2Name.TENSORCORE_UTIL_PCT.value:
                            metric_results[i].tensorcore_util_pct = metric[2]
                        elif family.name == MetricV2Name.MEMORY_BANDWIDTH_UTILIZATION.value:
                            metric_results[i].mem_bw_util_pct = metric[2]

            return metric_results

    except urllib.error.URLError as e:
        logging.log_first_n(logging.ERROR, "Failed to fetch metrics from %s: %s", 5, addr, e)
        return []
