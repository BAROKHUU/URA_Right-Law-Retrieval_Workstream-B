import pytest

from legal_retrieval.runtime_metrics import directory_size_bytes, latency_summary, resource_usage


def test_directory_size_bytes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"1234")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"123456")
    assert directory_size_bytes(tmp_path) == 10


def test_latency_summary_separates_first_and_warm_queries():
    summary = latency_summary([1.0, 0.2, 0.4])
    assert summary["count"] == 3
    assert summary["total_seconds"] == pytest.approx(1.6)
    assert summary["median_seconds"] == 0.4
    assert summary["first_query_seconds"] == 1.0
    assert summary["warm_mean_seconds"] == pytest.approx(0.3)
    assert summary["p95_seconds"] is not None


def test_empty_latency_summary_and_resource_schema():
    assert latency_summary([])["mean_seconds"] is None
    usage = resource_usage()
    assert set(usage) == {"peak_process_ram_bytes", "cuda_available", "gpu_devices"}
