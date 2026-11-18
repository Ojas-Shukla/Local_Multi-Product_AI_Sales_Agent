"""Stress tests for the standalone Codex adapter and the local-only Brain guard.

The application Brain no longer routes inference through Codex. Direct
CodexClient tests remain here for compatibility and process-cleanup coverage.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

import sellbird.codex as hc

# The Codex provider supports macOS and Linux only (see sellbird.codex).
pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="the Codex provider isn't supported on Windows")
from sellbird.config import BrainConfig
from tests.test_codex import _write_login

FAKE = str(Path(__file__).with_name("fake_codex.py"))


@pytest.fixture
def soak(monkeypatch, tmp_path):
    runtimes = tmp_path / "runtimes"
    runtimes.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(runtimes))
    native = tmp_path / "native"
    native.mkdir()
    _write_login(native)
    logs = tmp_path / "logs"
    logs.mkdir()
    real_exec = asyncio.create_subprocess_exec

    async def exec_routed(program, *args, **kwargs):
        # "codex@<scenario>" selects the fake's scenario per client.
        assert program.startswith("codex@"), program
        scenario = program.split("@", 1)[1]
        return await real_exec(sys.executable, FAKE, scenario, str(logs / f"{scenario}.log"),
                               *args, **kwargs)

    monkeypatch.setattr(hc.asyncio, "create_subprocess_exec", exec_routed)

    class Soak:
        def __init__(self):
            self.native = native
            self.runtimes = runtimes

        def entries(self, scenario=None):
            files = [logs / f"{scenario}.log"] if scenario else sorted(logs.glob("*.log"))
            out = []
            for f in files:
                if f.exists():
                    out += [json.loads(line) for line in f.read_text().splitlines()]
            return out

        def pids(self):
            return [e["pid"] for e in self.entries() if e.get("cmd") == "app-server"]

        def client(self, scenario):
            return hc.CodexClient(model="gpt-5.5", codex_path=f"codex@{scenario}",
                                  native_home=native)

    return Soak()


def _native_identity(home: Path):
    st = (home / "auth.json").stat()
    return ((home / "auth.json").read_bytes(), st.st_ino, st.st_mode, st.st_mtime_ns)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


def test_brain_rejects_the_legacy_codex_provider():
    """Codex runtime utilities remain testable, but Brain is Ollama-only."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Only local Ollama inference"):
        BrainConfig(provider="codex", model="gpt-5.5")


@pytest.mark.asyncio
async def test_soak_many_concurrent_clients(soak):
    """Six independent clients, five calls each, all at once."""
    clients = [soak.client("smart") for _ in range(6)]
    before = _native_identity(soak.native)
    started = time.monotonic()
    results = await asyncio.gather(*(
        c.run(f"Classify this email reply {i}-{n}: pricing?", timeout=30)
        for i, c in enumerate(clients) for n in range(5)))
    elapsed = time.monotonic() - started
    assert [r.text for r in results] == ["interested"] * 30
    assert len({r.thread_id for r in results}) == 30  # one runtime per call
    assert elapsed < 120
    if os.name == "posix":
        assert not [p for p in soak.pids() if _alive(p)]
    assert list(soak.runtimes.iterdir()) == []
    assert _native_identity(soak.native) == before


@pytest.mark.asyncio
async def test_soak_cancellation_storm(soak):
    """Cancel calls at many different points; nothing may leak."""
    client = soak.client("smart")
    before = _native_identity(soak.native)
    for delay in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
        task = asyncio.ensure_future(client.run("FAULT:hang", timeout=30))
        await asyncio.sleep(delay)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    # The client still works after all that.
    assert (await client.run("Classify this email reply: pricing?", timeout=30)).text == "interested"
    if os.name == "posix":
        assert not [p for p in soak.pids() if _alive(p)]
    assert list(soak.runtimes.iterdir()) == []
    assert _native_identity(soak.native) == before
