# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, payload-free diagnostics for UCXX communication frames."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from collections import deque
from typing import TextIO


class FrameTrace:
    def __init__(self, capacity: int = 1024):
        self._records: deque[str] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._pid = os.getpid()

    def record(
        self,
        stage: str,
        *,
        endpoint: int,
        sequence: int,
        frames=(),
        sizes=(),
        error: BaseException | None = None,
        part: str | None = None,
        tag: int | None = None,
        length: int | None = None,
    ) -> None:
        samples = []
        for frame in frames[:16]:
            if hasattr(frame, "__cuda_array_interface__"):
                samples.append((None, None, "cuda"))
            else:
                try:
                    view = memoryview(frame).cast("B")
                except (TypeError, ValueError):
                    samples.append((None, None, "unavailable"))
                else:
                    frame_size = len(view)
                    sampled_bytes = min(frame_size, 65536)
                    digest = hashlib.blake2b(
                        view[:sampled_bytes], digest_size=8
                    ).hexdigest()
                    samples.append((frame_size, sampled_bytes, digest))

        tag_text = f"0x{tag:x}" if tag is not None else None
        line = (
            f"UCXX_FRAME_TRACE time_ns={time.time_ns()} pid={os.getpid()} "
            f"ep=0x{endpoint:x} seq={sequence} stage={stage} "
            f"part={part} tag={tag_text} length={length} "
            f"samples={tuple(samples)} "
            f"sizes={tuple(sizes[:16])} count={len(sizes)} "
            f"error={type(error).__name__ if error is not None else None}"
        )
        with self._lock:
            if os.getpid() != self._pid:
                self._records.clear()
                self._pid = os.getpid()
            self._records.append(line)

    def dump(self, stream: TextIO | None = None) -> None:
        if stream is None:
            stream = sys.stderr
        with self._lock:
            records = tuple(self._records) if os.getpid() == self._pid else ()
        if records:
            print(
                f"UCXX_FRAME_TRACE_BEGIN pid={os.getpid()} count={len(records)}",
                file=stream,
            )
            for record in records:
                print(record, file=stream)
            print(f"UCXX_FRAME_TRACE_END pid={os.getpid()}", file=stream, flush=True)
