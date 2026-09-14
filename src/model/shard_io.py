"""Fixed-size training shards (CHAI v1). No TensorFlow."""

from __future__ import annotations

import mmap
import os
import struct
from typing import BinaryIO

import numpy as np

NAME = b"CHAI"
VERSION = 1
N_BOARD = 8 * 8 * 25
N_EXTRA = 19
N_WDL = 3
N_POLICY = 8 * 8 * 73
_HEADER_STRUCT = struct.Struct("<4sIIIII")
HEADER_SIZE = _HEADER_STRUCT.size

# measured in bytes
RECORD_SIZE = (N_BOARD + N_EXTRA + N_WDL + N_POLICY) * 4


def pack_header() -> bytes:
    return _HEADER_STRUCT.pack(NAME, VERSION, N_BOARD, N_EXTRA, N_WDL, N_POLICY)


def validate_header(raw: bytes) -> None:
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"truncated shard header ({len(raw)} bytes)")
    name, version, n_board, n_extra, n_wdl, n_policy = _HEADER_STRUCT.unpack(raw)
    if name != NAME:
        raise ValueError(f"bad shard type {name!r}")
    if version != VERSION:
        raise ValueError(f"unsupported shard version {version}")
    expected = (N_BOARD, N_EXTRA, N_WDL, N_POLICY)
    got = (n_board, n_extra, n_wdl, n_policy)
    if got != expected:
        raise ValueError(f"shard field sizes {got} != {expected}")


def shard_len(path: str) -> int:
    size = os.path.getsize(path)
    if size < HEADER_SIZE:
        return 0
    return (size - HEADER_SIZE) // RECORD_SIZE


def _as_f32(x, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} has {a.size} floats, expected {n}")
    return np.ascontiguousarray(a, dtype=np.float32)


def pack_record(board, extra, wdl, policy) -> bytes:
    return b"".join(
        (
            _as_f32(board, N_BOARD, "board").tobytes(),
            _as_f32(extra, N_EXTRA, "extra").tobytes(),
            _as_f32(wdl, N_WDL, "wdl").tobytes(),
            _as_f32(policy, N_POLICY, "policy").tobytes(),
        )
    )


def unpack_record(buf) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = bytes(buf) if not isinstance(buf, (bytes, bytearray)) else buf
    if len(raw) != RECORD_SIZE:
        raise ValueError(f"record has {len(raw)} bytes, expected {RECORD_SIZE}")
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    o = 0
    board = arr[o : o + N_BOARD].reshape(8, 8, 25)
    o += N_BOARD
    extra = arr[o : o + N_EXTRA]
    o += N_EXTRA
    wdl = arr[o : o + N_WDL]
    o += N_WDL
    policy = arr[o : o + N_POLICY]
    return board, extra, wdl, policy


class ShardWriter:
    def __init__(self, path: str):
        self.path = path
        self.n = 0
        self._closed = False
        self._f: BinaryIO = open(path, "wb")
        try:
            self._f.write(pack_header())
        except Exception:
            self._f.close()
            self._closed = True
            raise

    def write(self, board, extra, wdl, policy) -> None:
        if self._closed:
            raise ValueError(f"ShardWriter already closed: {self.path}")
        self._f.write(pack_record(board, extra, wdl, policy))
        self.n += 1

    def close(self) -> None:
        if self._closed:
            return
        self._f.flush()
        self._f.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ShardReader:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        validate_header(self._f.read(HEADER_SIZE))
        self.n = shard_len(path)
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if i < 0:
            i += self.n
        if i < 0 or i >= self.n:
            raise IndexError(i)
        start = HEADER_SIZE + i * RECORD_SIZE
        return unpack_record(self._mm[start : start + RECORD_SIZE])

    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        if mm is not None:
            mm.close()
            self._mm = None
        f = getattr(self, "_f", None)
        if f is not None:
            f.close()
            self._f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
