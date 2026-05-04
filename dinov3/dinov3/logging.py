# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Lightweight logging utilities.

The full DINOv3 repository provides a richer logging module. This trimmed
checkout only needs a minimal implementation to satisfy imports used by the
training/evaluation helpers.
"""

from __future__ import annotations

import datetime as _datetime
import logging as _logging
import time as _time
from collections import defaultdict, deque
from typing import Iterable, Iterator, Optional

import torch

logger = _logging.getLogger("dinov3")


class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size: int = 20, fmt: str = "{median:.4f} ({global_avg:.4f})") -> None:
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item() if len(d) else 0.0

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.mean().item() if len(d) else 0.0

    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self) -> float:
        return self.deque[-1] if self.deque else 0.0

    def synchronize_between_processes(self) -> None:
        # Minimal stub: no distributed sync in trimmed environment.
        return None

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter: str = "\t", output_file: Optional[str] = None) -> None:
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def __str__(self) -> str:
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def log_every(self, iterable: Iterable, print_freq: int, header: Optional[str] = None) -> Iterator:
        i = 0
        header = header or ""
        start_time = _time.time()
        end = start_time
        for obj in iterable:
            data_time = _time.time() - end
            yield obj
            iter_time = _time.time() - end
            if print_freq > 0 and i % print_freq == 0:
                eta_seconds = iter_time * (len(iterable) - i - 1) if hasattr(iterable, "__len__") else 0
                eta_string = str(_datetime.timedelta(seconds=int(eta_seconds)))
                log_msg = self.delimiter.join(
                    [
                        header,
                        f"[{i}/{len(iterable) if hasattr(iterable, '__len__') else '?'}]",
                        f"eta: {eta_string}",
                        str(self),
                        f"time: {iter_time:.4f}",
                        f"data: {data_time:.4f}",
                    ]
                )
                logger.info(log_msg)
                if self.output_file:
                    try:
                        with open(self.output_file, "a", encoding="utf-8") as f:
                            f.write(log_msg + "\n")
                    except OSError:
                        pass
            i += 1
            end = _time.time()
        total_time = _time.time() - start_time
        total_time_str = str(_datetime.timedelta(seconds=int(total_time)))
        logger.info(f"{header} Total time: {total_time_str}")


def setup_logging() -> None:
    """Minimal setup hook for compatibility."""
    if not _logging.getLogger().handlers:
        _logging.basicConfig(level=_logging.INFO)
