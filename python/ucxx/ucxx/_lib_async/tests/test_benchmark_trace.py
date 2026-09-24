# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path


def _trace_class():
    backend_path = (
        Path(__file__).resolve().parents[2] / "benchmarks/backends/ucxx_async.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ucxx_async_trace_under_test", backend_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "_BenchmarkTrace")
    return module._BenchmarkTrace


def test_benchmark_trace_is_opt_in(monkeypatch, capsys):
    monkeypatch.delenv("UCXX_BENCH_TRACE", raising=False)
    trace = _trace_class()("client")
    trace.record("recv-start", 0, 42)
    trace.dump()
    assert capsys.readouterr().err == ""


def test_benchmark_trace_keeps_only_recent_transitions(monkeypatch, capsys):
    monkeypatch.setenv("UCXX_BENCH_TRACE", "1")
    trace = _trace_class()("server", capacity=2)
    trace.record("old-stage", 0, 42)
    trace.record("send-done", 1, 42)
    trace.record("close-start", 1, 42)
    trace.dump()
    output = capsys.readouterr().err
    assert "old-stage" not in output
    assert "send-done" in output
    assert "close-start" in output
    assert "role=server" in output
    assert "ep=0x2a" in output
