from training_framework.trainer import cuda_memory_metrics


class CpuDevice:
    type = "cpu"


def test_cuda_memory_metrics_returns_empty_for_cpu_device():
    assert cuda_memory_metrics(CpuDevice()) == {}
