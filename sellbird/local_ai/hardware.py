"""Hardware detection and local-model sizing recommendations.

Uses operating-system facilities and optional NVIDIA tooling only. No network
calls are made. Recommendations are advisory; the configured assignment model
remains qwen3:1.7b unless the operator explicitly changes OLLAMA_MODEL.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareReport:
    operating_system: str
    architecture: str
    cpu_logical_cores: int
    cpu_physical_cores: int | None
    ram_total_gb: float | None
    gpu_name: str | None
    gpu_vram_gb: float | None
    gpu_detection: str
    profile: str
    recommended_model: str
    recommended_configuration: str
    recommended_settings: dict
    configured_model: str = "qwen3:1.7b"

    def to_dict(self) -> dict:
        return asdict(self)


def _linux_ram_bytes() -> int | None:
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _ram_bytes() -> int | None:
    system = platform.system()
    if system == "Linux":
        return _linux_ram_bytes()
    if system == "Darwin":
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2)
            return int(output.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    if system == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            return None
    return None


def _physical_cores() -> int | None:
    if platform.system() == "Linux":
        try:
            text = open("/proc/cpuinfo", encoding="utf-8").read()
            pairs = set(re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", text, re.S))
            if pairs:
                return len(pairs)
        except OSError:
            pass
    return None


def _nvidia_gpu() -> tuple[str | None, float | None, str]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, None, "not_detected_by_nvidia_smi"
    try:
        output = subprocess.check_output(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=3, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if not output:
            return None, None, "nvidia_smi_no_devices"
        name, _, memory = output[0].partition(",")
        vram_mb = float(memory.strip())
        return name.strip(), round(vram_mb / 1024, 2), "detected_nvidia"
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None, None, "nvidia_smi_unavailable_or_error"


def _other_gpu() -> tuple[str | None, float | None, str]:
    """Best-effort GPU detection for ROCm, macOS, and Windows hosts."""
    system = platform.system()
    if system == "Linux" and shutil.which("rocm-smi"):
        try:
            raw = subprocess.check_output(
                [shutil.which("rocm-smi"), "--showproductname", "--showmeminfo", "vram", "--json"],
                text=True, timeout=4, stderr=subprocess.DEVNULL,
            )
            payload = json.loads(raw)
            flat = []
            def walk(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        flat.append((str(key).lower(), child))
                        walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)
            walk(payload)
            name = next((str(v) for k, v in flat if "product name" in k or "card series" in k), None)
            memory = next((v for k, v in flat if "vram total memory" in k or "total vram" in k), None)
            vram_gb = round(float(re.sub(r"[^0-9.]", "", str(memory))) / (1024 ** 3), 2) if memory else None
            return name, vram_gb, "detected_rocm" if name else "rocm_smi_no_devices"
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            return None, None, "rocm_smi_unavailable_or_error"
    if system == "Darwin" and shutil.which("system_profiler"):
        try:
            raw = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                text=True, timeout=5, stderr=subprocess.DEVNULL,
            )
            displays = json.loads(raw).get("SPDisplaysDataType", [])
            if displays:
                item = displays[0]
                name = item.get("sppci_model") or item.get("_name")
                memory = item.get("spdisplays_vram") or item.get("spdisplays_vram_shared")
                match = re.search(r"([0-9.]+)\s*(GB|MB)", str(memory), re.I) if memory else None
                vram_gb = (float(match.group(1)) * (1 if match.group(2).upper() == "GB" else 1 / 1024)) if match else None
                return name, round(vram_gb, 2) if vram_gb is not None else None, "detected_macos"
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            return None, None, "system_profiler_unavailable_or_error"
    if system == "Windows" and shutil.which("powershell"):
        command = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
        )
        try:
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", command],
                text=True, timeout=5, stderr=subprocess.DEVNULL,
            )
            payload = json.loads(raw)
            item = payload[0] if isinstance(payload, list) and payload else payload
            if isinstance(item, dict):
                memory = item.get("AdapterRAM")
                vram_gb = round(int(memory) / (1024 ** 3), 2) if memory else None
                return item.get("Name"), vram_gb, "detected_windows"
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            return None, None, "powershell_gpu_query_failed"
    return None, None, "not_detected_by_available_gpu_tools"


def _gpu_resources() -> tuple[str | None, float | None, str]:
    nvidia = _nvidia_gpu()
    if nvidia[0] or nvidia[1]:
        return nvidia
    other = _other_gpu()
    if other[0] or other[1]:
        return other
    return other


def detect_hardware(configured_model: str = "qwen3:1.7b") -> HardwareReport:
    """Detect available host resources and recommend a practical model tier."""
    ram = _ram_bytes()
    ram_gb = round(ram / (1024 ** 3), 2) if ram else None
    gpu_name, gpu_vram, gpu_detection = _gpu_resources()
    # High-performance tier is intentionally conservative: enough system RAM
    # and a dedicated GPU with usable VRAM for a larger quantized model.
    high_performance = bool(ram_gb is not None and ram_gb >= 32 and gpu_vram is not None and gpu_vram >= 8)
    if high_performance:
        profile = "high_performance"
        recommended_model = "qwen3:8b"
        recommendation = "32+ GB RAM and 8+ GB NVIDIA VRAM detected; larger quantized local models are practical."
        recommended_settings = {
            "model": "qwen3:8b", "num_ctx": 8192, "num_predict": 1024,
            "temperature": 0.2, "keep_alive": "10m",
        }
    else:
        profile = "standard"
        recommended_model = "qwen3:1.7b"
        recommendation = "Use the lightweight qwen3:1.7b configuration for ordinary laptops or CPU-only systems."
        recommended_settings = {
            "model": "qwen3:1.7b", "num_ctx": 4096, "num_predict": 512,
            "temperature": 0.2, "keep_alive": "5m",
        }
    return HardwareReport(
        operating_system=platform.platform(),
        architecture=platform.machine(),
        cpu_logical_cores=os.cpu_count() or 1,
        cpu_physical_cores=_physical_cores(),
        ram_total_gb=ram_gb,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        gpu_detection=gpu_detection,
        profile=profile,
        recommended_model=recommended_model,
        recommended_configuration=recommendation,
        recommended_settings=recommended_settings,
        configured_model=configured_model,
    )
