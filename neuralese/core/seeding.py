"""Stable, cross-process deterministic seed derivation.

Never uses Python's randomized hash() for experimental seeds.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Union


def stable_seed(master_seed: int, *parts: Union[str, int]) -> int:
    """Derive a 32-bit seed from master_seed and ordered parts.

    Algorithm: SHA-256 over UTF-8 encoding of ``master_seed|part0|part1|...``,
    then take the first 4 bytes as a big-endian unsigned int, modulo 2**31-1
    for numpy Generator compatibility.

    Identical inputs always yield identical outputs across processes and
    interpreter runs (independent of PYTHONHASHSEED).
    """
    payload = str(int(master_seed))
    for p in parts:
        payload += "|" + str(p)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    value = struct.unpack(">I", digest[:4])[0]
    return int(value % (2**31 - 1))


def task_index_seed(master_seed: int, task_index: int, role: str = "intervention") -> int:
    """Seed for a task by its stable index (not process-dependent id())."""
    return stable_seed(master_seed, role, int(task_index))
