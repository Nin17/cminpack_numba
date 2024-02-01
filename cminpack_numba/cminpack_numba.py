"""_summary_
"""
# pylint: disable=R0913, W0613
from __future__ import annotations

import warnings
from ctypes.util import find_library

import numpy as np
from llvmlite import binding
from numba import extending, njit, types

from .utils import ptr_from_val, val_from_ptr, get_extension_path

__all__ = [
    "enorm",
    "enorm_",
    # "chkder",
    # "dpmpar"
    # "sdpmpar",
    "hybrd1",
    "hybrd1_",
    "hybrd",
    "hybrd_",
    "hybrj1",
    "hybrj1_",
    # "hybrj",
    # "hybrj_",
    "lmdif1",
    "lmdif1_",
    "lmdif",
    "lmdif_",
    "lmder1",
    "lmder1_",
    # "lmder",
    # "lmder_",
    "lmstr1",
    "lmstr1_",
    # "lmstr",
    # "lmstr_",
]

def _check_dtype(args, dtype, error=True):
    if not all(i.dtype is dtype for i in args):
        if error:
            raise ValueError("All array arguments must be of the same dtype")
        warnings.warn("All array arguments should be of the same dtype")


def ensure_cminpack(dtype: str = "") -> None:
    """_summary_

    Parameters
    ----------
    dtype : str, optional
        _description_, by default ""

    Raises
    ------
    ImportError
        _description_
    """
    if find_library(f"cminpack{dtype}") is None and get_extension_path(f"libcminpack{dtype}") is None:
        raise ImportError(f"cminpack{dtype} library not found")


# Load double and single precision libraries
try:
    ensure_cminpack()
    binding.load_library_permanently(find_library("cminpack") or get_extension_path("libcminpack"))
except ImportError:
    warnings.warn(
        "cminpack not found. Double precision functions unavailable.")
try:
    ensure_cminpack("s")
    binding.load_library_permanently(find_library("cminpacks") or get_extension_path("libcminpacks"))
except ImportError:
    warnings.warn(
        "cminpacks not found. Single precision functions unavailable.")


_cminpack_prefix = {types.float32: "s", types.float64: ""}


def _cminpack_func(func, dtype):
    return f"{_cminpack_prefix[dtype]}{func}"


class Cminpack:
    """
    External functions from the cminpack(s) library.
    """

    # TODO chkder
    # TODO dpmpar
    # TODO hybrd
    # TODO hybrj
    # TODO lmdif
    # TODO lmder
    # TODO lmstr
    
    def __init__(self):
        FAILED = 0
        try:
            ensure_cminpack()
        except ImportError:
            FAILED += 1
        try:
            ensure_cminpack("s")
        except ImportError:
            FAILED += 1
        if FAILED == 2:
            raise ImportError("cminpack not found. Double and single precision functions unavailable.")

    @staticmethod
    def enorm(dtype: types.Float) -> types.ExternalFunction:  # pylint: disable= C0116
        sig = dtype(
            types.int32,  # n
            types.CPointer(dtype),  # *x
        )
        return types.ExternalFunction(_cminpack_func("enorm", dtype), sig)

    @staticmethod
    def hybrd(dtype: types.Float) -> types.ExternalFunction:  # pylint: disable= C0116
        sig = types.int32(
            types.long_,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            dtype,  # xtol
            types.int32,  # maxfev
            types.int32,  # ml
            types.int32,  # mu
            dtype,  # epsfcn
            types.CPointer(dtype),  # *diag
            types.int32,  # mode
            dtype,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *nfev
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            types.CPointer(dtype),  # *r
            types.int32,  # lr
            types.CPointer(dtype),  # *qtf
            types.CPointer(dtype),  # *wa1
            types.CPointer(dtype),  # *wa2
            types.CPointer(dtype),  # *wa3
            types.CPointer(dtype),  # *wa4
        )
        return types.ExternalFunction(_cminpack_func("hybrd", dtype), sig)

    @staticmethod
    def hybrd1(dtype: types.Float) -> types.ExternalFunction:  # pylint: disable= C0116
        sig = types.int32(
            types.long_,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            dtype,  # tol
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(_cminpack_func("hybrd1", dtype), sig)

    @staticmethod
    def hybrj1(dtype: types.Float) -> types.ExternalFunction:  # pylint: disable= C0116
        sig = types.int32(
            types.long_,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            dtype,  # tol
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(f"{_cminpack_prefix[dtype]}hybrj1", sig)

    @staticmethod
    def lmdif(dtype: types.Float) -> types.ExternalFunction:  # pylint: disable= C0116
        sig = types.int32(
            types.long_,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            dtype,  # ftol
            dtype,  # xtol
            dtype,  # gtol
            types.int32,  # maxfev
            dtype,  # epsfcn
            types.CPointer(dtype),  # *diag
            types.int32,  # mode
            dtype,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *nfev
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(dtype),  # *qtf
            types.CPointer(dtype),  # *wa1
            types.CPointer(dtype),  # *wa2
            types.CPointer(dtype),  # *wa3
            types.CPointer(dtype),  # *wa4
        )
        return types.ExternalFunction(f"{_cminpack_prefix[dtype]}lmdif", sig)

    @staticmethod
    def lmdif1(dtype: types.Float) -> types.ExternalFunction:  # pylint: disable= C0116
        sig = types.int32(
            types.long_,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            dtype,  # tol
            types.CPointer(types.int32),  # *iwa
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(f"{_cminpack_prefix[dtype]}lmdif1", sig)

    @staticmethod
    def lmder1(dtype: types.Float) -> types.ExternalFunction:  # pylint: disable= C0116
        sig = types.int32(
            types.long_,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            dtype,  # tol
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(f"{_cminpack_prefix[dtype]}lmder1", sig)

    @staticmethod
    def lmstr1(dtype: types.Float) -> types.ExternalFunction:  # pylint: disable= C0116
        sig = types.int32(
            types.long_,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            dtype,  # tol
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(f"{_cminpack_prefix[dtype]}lmstr1", sig)


# --------------------------------------- enorm -------------------------------------- #


def _enorm(n, x):
    raise NotImplementedError


@extending.overload(_enorm)
def _enorm_overload(n, x):
    _enorm_cfunc = Cminpack().enorm(x.dtype)

    def impl(n, x):
        assert x.ndim == 1
        return _enorm_cfunc(n, x.ctypes)

    return impl


@njit
def enorm_(n, x):
    return _enorm(n, x)


@njit
def enorm(x):
    return _enorm(x.size, x)


# -------------------------------------- hybrd1 -------------------------------------- #


def _hybrd1(fcn, n, x, fvec, tol, wa, lwa, udata):
    raise NotImplementedError


@extending.overload(_hybrd1)
def _hybrd1_overload(fcn, n, x, fvec, tol, wa, lwa, udata):
    _check_dtype((fvec, wa), x.dtype)
    _hybrd1_cfunc = Cminpack().hybrd1(x.dtype)

    @extending.register_jitable
    def impl(fcn, n, x, fvec, tol, wa, lwa, udata):
        info = _hybrd1_cfunc(
            fcn, udata.ctypes, n, x.ctypes, fvec.ctypes, tol, wa.ctypes, lwa
        )
        return x, fvec, info

    if udata is not types.none:
        return impl
    return lambda fcn, n, x, fvec, tol, wa, lwa, udata: impl(
        fcn, n, x, fvec, tol, wa, lwa, np.array(0, dtype=np.bool_)
    )


@njit
def hybrd1_(fcn, n, x, fvec, tol, wa, lwa, udata=None):
    return _hybrd1(fcn, n, x, fvec, tol, wa, lwa, udata)


@njit
def _hybrd1_args(x):
    n = np.int32(x.size)
    lwa = np.int32((n * (3 * n + 13)) // 2)  # TODO check this
    fvec = np.empty(n, dtype=x.dtype)
    wa = np.empty(lwa, dtype=x.dtype)
    return n, lwa, fvec, wa


@njit
def hybrd1(fcn, x, tol=None, udata=None):
    tol = 1.49012e-8 if tol is None else tol
    n, lwa, fvec, wa = _hybrd1_args(x)
    return _hybrd1(fcn, n, x.copy(), fvec, tol, wa, lwa, udata)


# --------------------------------------- hybrd -------------------------------------- #


def _hybrd(fcn, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor, nprint,
           nfev, fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4, udata):
    raise NotImplementedError


@extending.overload(_hybrd)
def _hybrd_overload(fcn, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor,
                    nprint, nfev, fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4, udata):
    _check_dtype((fvec, fjac, qtf, wa1, wa2, wa3, wa4), x.dtype)
    _hybrd_cfunc = Cminpack().hybrd(x.dtype)

    @extending.register_jitable
    def impl(fcn, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor, nprint,
             nfev, fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4, udata):
        return _hybrd_cfunc(
            fcn, udata.ctypes, n, x.ctypes, fvec.ctypes, xtol, maxfev, ml, mu, epsfcn,
            diag.ctypes, mode, factor, nprint, nfev, fjac.ctypes, ldfjac, r.ctypes,
            lr, qtf.ctypes, wa1.ctypes, wa2.ctypes, wa3.ctypes, wa4.ctypes)

    if udata is not types.none:
        return impl
    return (lambda fcn, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor, nprint,
            nfev, fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4, udata: impl(
                fcn, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor, nprint,
                nfev, fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4, np.array(0, dtype=np.bool_))
            )


@njit
def hybrd_(fcn, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor, nprint,
           nfev, fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4, udata=None):
    return _hybrd(fcn, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor, nprint,
                  nfev, fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4, udata)


@njit
def _hybrd_args(x):
    n = np.int32(x.size)
    fvec = np.empty(n, dtype=x.dtype)
    nfevptr = ptr_from_val(np.int32(0))
    fjac = np.empty((n, n), dtype=x.dtype)
    ldfjac = n
    lr = (n*(n+1))//2
    r = np.empty(lr, dtype=x.dtype)
    qtf = np.empty(n, dtype=x.dtype)
    wa = np.empty(4*n, dtype=x.dtype)
    wa1 = wa[:n]
    wa2 = wa[n:2*n]
    wa3 = wa[2*n:3*n]
    wa4 = wa[3*n:]

    return n, fvec, nfevptr, fjac, ldfjac, lr, r, qtf, wa1, wa2, wa3, wa4


@njit
def hybrd(fcn, x, xtol=1.49012e-8, maxfev=0, ml=None, mu=None, epsfcn=None, diag=None,
          mode=1, factor=100., nprint=0, udata=None):
    n, fvec, nfevptr, fjac, ldfjac, lr, r, qtf, wa1, wa2, wa3, wa4 = _hybrd_args(
        x)
    if diag is None:
        diag = np.ones(n, dtype=x.dtype)
    if epsfcn is None:
        epsfcn = np.finfo(x.dtype).eps
    if not maxfev:
        maxfev = 200 * (n + 1)
    if ml is None:
        ml = n
    if mu is None:
        mu = n
    x0 = x.copy()
    info = _hybrd(fcn, n, x0, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor,
                  nprint, nfevptr, fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4, udata)
    nfev = val_from_ptr(nfevptr)
    return x0, fvec, info, fjac, r, qtf, nfev

# -------------------------------------- hybrj1 -------------------------------------- #


def _hybrj1(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata):
    ...


@extending.overload(_hybrj1)
def _hybrj1_overload(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata):
    _check_dtype((fvec, fjac, wa), x.dtype)
    _hybrj1_cfunc = Cminpack().hybrj1(x.dtype)

    @extending.register_jitable
    def impl(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata):
        info = _hybrj1_cfunc(
            fcn,
            udata.ctypes,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            tol,
            wa.ctypes,
            lwa,
        )
        return x, fvec, fjac, info

    if udata is not types.none:
        return impl
    return lambda fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata: impl(
        fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, np.array(
            0, dtype=np.bool_)
    )


@njit
def hybrj1_(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata=None):
    return _hybrj1(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata)


@njit
def _hybrj1_args(x):
    # ??? do i need the int32 here?
    n = np.int32(x.size)
    lwa = np.int32((n * (3 * n + 13)) // 2)
    fvec = np.empty(n, dtype=x.dtype)
    fjac = np.empty((n, n), dtype=x.dtype)
    wa = np.empty(lwa, dtype=x.dtype)
    return n, lwa, fvec, fjac, wa


@njit
def hybrj1(fcn, x, tol=None, udata=None):
    tol = 1.49012e-8 if tol is None else tol
    n, lwa, fvec, fjac, wa = _hybrj1_args(x)
    return _hybrj1(fcn, n, x.copy(), fvec, fjac, n, tol, wa, lwa, udata)


# --------------------------------------- hybrj -------------------------------------- #


# TODO hybrj


# -------------------------------------- lmdif1 -------------------------------------- #


def _lmdif1(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata):
    raise NotImplementedError


@extending.overload(_lmdif1)
def _lmdif1_overload(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata):
    _check_dtype((fvec, wa), x.dtype)
    _lmdif1_cfunc = Cminpack().lmdif1(x.dtype)

    @extending.register_jitable
    def impl(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata):
        info = _lmdif1_cfunc(
            fcn,
            udata.ctypes,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            tol,
            iwa.ctypes,
            wa.ctypes,
            lwa,
        )
        return info

    if udata is not types.none:
        return impl
    return lambda fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata: impl(
        fcn, m, n, x, fvec, tol, iwa, wa, lwa, np.array(0, dtype=np.bool_)
    )


@njit
def lmdif1_(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata=None):
    return _lmdif1(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata)


@njit
def _lmdif1_args(m, x):
    # ??? do i need the int32 here?
    n = np.int32(x.size)
    lwa = np.int32(m * n + 5 * n + m)
    fvec = np.empty(m, dtype=x.dtype)
    wa = np.empty(lwa, dtype=x.dtype)
    iwa = np.empty(n, dtype=np.int32)
    return n, lwa, fvec, wa, iwa


@njit
def lmdif1(fcn, m, x, tol=None, udata=None):
    tol = 1.49012e-8 if tol is None else tol
    n, lwa, fvec, wa, iwa = _lmdif1_args(m, x)
    x0 = x.copy()
    info = _lmdif1(fcn, m, n, x0, fvec, tol, iwa, wa, lwa, udata)
    return x0, fvec, info


# --------------------------------------- lmdif -------------------------------------- #


def _lmdif(fcn, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor,
           nprint, nfev, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4, udata
           ):
    raise NotImplementedError


@extending.overload(_lmdif)
def _lmdif1_overload(
    fcn, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor,
    nprint, nfev, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4, udata
):
    _check_dtype((fvec, fjac, qtf, wa1, wa2, wa3, wa4), x.dtype)
    _lmdif_cfunc = Cminpack().lmdif(x.dtype)

    @extending.register_jitable
    def impl(
        fcn, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor,
        nprint, nfev, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4, udata
    ):
        info = _lmdif_cfunc(
            fcn, udata.ctypes, m, n, x.ctypes, fvec.ctypes, ftol, xtol, gtol, maxfev,
            epsfcn, diag.ctypes, mode, factor, nprint, nfev, fjac.ctypes, ldfjac,
            ipvt.ctypes, qtf.ctypes, wa1.ctypes, wa2.ctypes, wa3.ctypes, wa4.ctypes
        )
        return info

    if udata is not types.none:
        return impl
    return (lambda fcn, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor,
            nprint, nfev, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4, udata: impl(
                fcn, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor,
                nprint, nfev, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4, np.array(
                    0, dtype=np.bool_)
            ))


@njit
def lmdif_(fcn, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor,
           nprint, nfev, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4, udata=None):
    return _lmdif(fcn, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor,
                  nprint, nfev, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4, udata)


@njit
def _lmdif_args(m, x):
    # ??? do i need the int32 here?
    n = np.int32(x.size)
    # lwa = np.int32(m * n + 5 * n + m)
    fvec = np.empty(m, dtype=x.dtype)
    # m x n seems to work in row-major order
    fjac = np.empty((m, n), dtype=x.dtype)
    nfevptr = ptr_from_val(np.int32(0))
    ldfjac = m
    ipvt = np.empty(n, dtype=np.int32)
    qtf = np.empty(n, dtype=x.dtype)
    wa = np.empty(3*n + m, dtype=x.dtype)
    wa1 = wa[:n]
    wa2 = wa[n:2*n]
    wa3 = wa[2*n:3*n]
    wa4 = wa[3*n:]
    return n, fvec, nfevptr, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4


@njit
def lmdif(fcn, m, x, ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0, maxfev=None, epsfcn=None,
          diag=None, mode=1, factor=100., nprint=0, udata=None):
    n, fvec, nfevptr, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4 = _lmdif_args(
        m, x)
    # if mode == 2 and diag is None:
    #     diag = np.ones(n, dtype=x.dtype)
    # elif diag is None:
    #     diag = np.empty(1, dtype=x.dtype)
    if diag is None:
        diag = np.ones(n, dtype=x.dtype)
    if epsfcn is None:
        epsfcn = np.finfo(x.dtype).eps
    if not maxfev:
        maxfev = 200 * (n + 1)
    x0 = x.copy()
    info = _lmdif(fcn, m, n, x0, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor,
                  nprint, nfevptr, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4, udata)
    nfev = val_from_ptr(nfevptr)
    return x, fvec, info, fjac, ipvt, qtf, nfev
# -------------------------------------- lmder1 -------------------------------------- #


def _lmder1(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata):
    ...


@extending.overload(_lmder1)
def _lmder1_overload(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata):
    _check_dtype((fvec, fjac, wa), x.dtype)
    _lmder1_cfunc = Cminpack().lmder1(x.dtype)

    @extending.register_jitable
    def impl(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata):
        info = _lmder1_cfunc(
            fcn,
            udata.ctypes,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            tol,
            ipvt.ctypes,
            wa.ctypes,
            lwa,
        )
        return x, fvec, fjac, info

    if udata is not types.none:
        return impl
    return lambda fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata: impl(
        fcn,
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        tol,
        ipvt,
        wa,
        lwa,
        np.array(0, dtype=np.bool_),
    )


@njit
def lmder1_(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata=None):
    return _lmder1(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata)


@njit
def _lmder1_args(m, x):
    # ??? do i need the int32 here?
    n = np.int32(x.size)
    lwa = np.int32(5 * n + m)
    fvec = np.empty(m, dtype=x.dtype)
    fjac = np.empty((m, n), dtype=x.dtype)
    wa = np.empty(lwa, dtype=x.dtype)
    ipvt = np.empty(n, dtype=np.int32)
    return n, lwa, fvec, fjac, wa, ipvt


@njit
def lmder1(fcn, m, x, tol=None, udata=None):
    tol = 1.49012e-8 if tol is None else tol
    n, lwa, fvec, fjac, wa, ipvt = _lmder1_args(m, x)
    return lmder1_(fcn, m, n, x.copy(), fvec, fjac, n, tol, ipvt, wa, lwa, udata)


# --------------------------------------- lmder -------------------------------------- #


# TODO lmder


# -------------------------------------- lmstr1 -------------------------------------- #


def _lmstr1(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata):
    ...


@extending.overload
def _lmstr1_overload(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata):
    _check_dtype((fvec, fjac, wa), x.dtype)
    _lmstr1_cfunc = Cminpack().lmstr1(x.dtype)

    @extending.register_jitable
    def impl(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata):
        info = _lmstr1_cfunc(
            fcn,
            udata.ctypes,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            tol,
            ipvt.ctypes,
            wa.ctypes,
            lwa,
        )
        return x, fvec, fjac, info

    if udata is types.none:
        return impl

    return lambda fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata: impl(
        fcn,
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        tol,
        ipvt,
        wa,
        lwa,
        np.array(0, dtype=np.bool_),
    )


@njit
def lmstr1_(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata=None):
    return _lmstr1(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata)


@njit
def _lmstr1_args(m, x):
    # ??? do i need the int32 here?
    n = np.int32(x.size)
    lwa = np.int32(5 * n + m)
    fvec = np.empty(m, dtype=x.dtype)
    fjac = np.empty((m, n), dtype=x.dtype)
    wa = np.empty(lwa, dtype=x.dtype)
    ipvt = np.empty(n, dtype=np.int32)
    return n, lwa, fvec, fjac, wa, ipvt


@njit
def lmstr1(fcn, m, x, tol=None, udata=None):
    tol = 1.49012e-8 if tol is None else tol
    n, lwa, fvec, fjac, wa, ipvt = _lmstr1_args(m, x)
    return lmstr1_(fcn, m, n, x, fvec, fjac, n, tol, ipvt, wa, lwa, udata)


# --------------------------------------- lmstr -------------------------------------- #


# TODO lmstr
