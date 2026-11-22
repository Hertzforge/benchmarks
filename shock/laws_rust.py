"""followed and followed-hold from the Rust law math (hertzforge-mimo-shock-law-math, followed.rs) through the
closed-loop library in followed_rs/ (ctypes), as harness.Law, pulse by pulse.

    SHOCK_FOLLOWED_LIB=<path to libshock_followed.dylib>   (default: the worktree's build folder)

The setup is the harness's: the identification `frf`, trusted on the drive band (so the law's drive mask is the
harness's); probes at -40 and -34 dB, as the Python ports'. lib_info() gives the library's path and SHA-256.
"""
import ctypes as C
import hashlib
import json
import os

import numpy as np

from harness import Law

DEFAULT_LIB = os.path.expanduser('~/dev/target.noindex/sds-mshock-followed-wt/release/libshock_followed.dylib')
_lib = None
P = C.POINTER(C.c_double)


def lib_path():
    return os.environ.get('SHOCK_FOLLOWED_LIB', DEFAULT_LIB)


def lib_info():
    p = lib_path()
    return dict(path=p, sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest())


def lib():
    global _lib
    if _lib is None:
        L = C.CDLL(lib_path())
        d, sz, vp = C.c_double, C.c_size_t, C.c_void_p
        for name, res, args in (
                ('fl_last_error', sz, [C.c_char_p, sz]),
                ('fl_new', vp, [C.c_char_p, d, sz, sz, sz, d, d, P, P, sz, d, P, P, C.POINTER(C.c_uint8)]),
                ('fl_free', None, [vp]),
                ('fl_seed', C.c_int32, [vp, d, P]),
                ('fl_iterate', C.c_int32, [vp, sz, d, d, P, P, P]),
                ('fl_report', sz, [vp, C.c_char_p, sz])):
            f = getattr(L, name); f.restype, f.argtypes = res, args
        _lib = L
    return _lib


def _p(a):
    return a.ctypes.data_as(P)


def _err():
    b = C.create_string_buffer(4096); lib().fl_last_error(b, 4096); return b.value.decode()


class RustFollowed(Law):
    probes = (-40.0, -34.0)

    def __init__(self, kind):
        self.name, self.kind, self._h = kind, kind, None

    def ladder(self, table):
        return list(self.probes) + list(table)

    def start(self, s):
        self.s = s
        self._keep = [np.ascontiguousarray(x, np.float64) for x in (s.reference, s.natural_hz, s.points)]
        self._keep.append(np.ascontiguousarray(s.frf, np.complex128).view(np.float64).reshape(-1).copy())
        tr = np.ascontiguousarray(s.drive_mask(), np.uint8)
        ref, nat, pts, frf = self._keep
        self._h = lib().fl_new(self.kind.encode(), float(s.fs), s.n, s.nc, s.nd, float(s.band_lo), float(s.band_hi),
                               _p(ref), _p(nat), nat.size, float(s.q), _p(pts), _p(frf),
                               tr.ctypes.data_as(C.POINTER(C.c_uint8)))
        if not self._h:
            raise RuntimeError(_err())

    def _out(self, rc, out):
        if rc != 0:
            raise RuntimeError(_err())
        return out

    def seed(self, level):
        out = np.zeros((self.s.nd, self.s.n))
        return self._out(lib().fl_seed(self._h, float(level), _p(out)), out)

    def iterate(self, p):
        d = np.ascontiguousarray(p.drives, np.float64); y = np.ascontiguousarray(p.responses, np.float64)
        out = np.zeros((self.s.nd, self.s.n))
        return self._out(lib().fl_iterate(self._h, p.index, float(p.level_db), float(p.next_level_db), _p(d), _p(y),
                                          _p(out)), out)

    def report(self):
        if not self._h:
            return {}
        n = lib().fl_report(self._h, None, 0)
        b = C.create_string_buffer(n + 1); lib().fl_report(self._h, b, n + 1)
        r = json.loads(b.value.decode())
        return dict(probe_checks=r['probe_checks'], holds=r['holds'], last_values=r['values'])

    def __del__(self):
        if getattr(self, '_h', None):
            lib().fl_free(self._h); self._h = None


LAWS = {'followed': lambda: RustFollowed('followed'), 'followed-hold': lambda: RustFollowed('followed-hold')}
