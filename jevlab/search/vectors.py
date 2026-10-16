"""Half-precision embedding cache.

bge-m3 vectors were stored as float32, 4096 bytes each. New vectors are float16, half
that size. A 4096-byte blob is still read as float32 so a cache can be converted
while searches are using it.
"""

from __future__ import annotations

import numpy as np

# Full-precision bge-m3. This cache has never stored another width at this size.
LEGACY_F32_BYTES = 4096


def pack_half(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float16).tobytes()


def unpack_vec(blob: bytes) -> np.ndarray:
    """Float32 array, copied off the SQLite buffer."""
    raw = bytes(blob)
    if len(raw) == LEGACY_F32_BYTES:
        return np.array(np.frombuffer(raw, dtype=np.float32), dtype=np.float32, copy=True)
    return np.array(np.frombuffer(raw, dtype=np.float16), dtype=np.float32)
