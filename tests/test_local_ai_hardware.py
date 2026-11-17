from sellbird.local_ai import hardware


def test_standard_hardware_recommends_assignment_model(monkeypatch):
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(hardware, "_nvidia_gpu", lambda: (None, None, "not_detected"))
    monkeypatch.setattr(hardware, "_other_gpu", lambda: (None, None, "not_detected_by_available_gpu_tools"))
    report = hardware.detect_hardware()
    assert report.profile == "standard"
    assert report.recommended_model == "qwen3:1.7b"
    assert report.configured_model == "qwen3:1.7b"
    assert report.cpu_logical_cores >= 1


def test_high_performance_hardware_recommends_larger_tier(monkeypatch):
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(hardware, "_nvidia_gpu", lambda: ("Test GPU", 16.0, "detected_nvidia"))
    report = hardware.detect_hardware(configured_model="qwen3:1.7b")
    assert report.profile == "high_performance"
    assert report.recommended_model == "qwen3:8b"
    # Hardware advice never silently changes the assignment's configured model.
    assert report.configured_model == "qwen3:1.7b"


def test_report_serializes_for_dashboard(monkeypatch):
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: None)
    monkeypatch.setattr(hardware, "_nvidia_gpu", lambda: (None, None, "not_detected"))
    monkeypatch.setattr(hardware, "_other_gpu", lambda: (None, None, "not_detected_by_available_gpu_tools"))
    payload = hardware.detect_hardware().to_dict()
    assert payload["ram_total_gb"] is None
    assert payload["gpu_detection"] == "not_detected_by_available_gpu_tools"
