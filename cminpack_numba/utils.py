"""_summary_
"""

import ctypes as ct
from pathlib import Path

from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

__all__ = [
    "check_cfunc",
]


@intrinsic
def ptr_from_val(typingctx, data):
    # From: https://stackoverflow.com/a/59538114/15456681
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(data)(data)
    return sig, impl


@intrinsic
def val_from_ptr(typingctx, data):
    # From: https://stackoverflow.com/a/59538114/15456681
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = data.dtype(data)
    return sig, impl


def check_cfunc(func, *args):
    # TODO documentation
    """_summary_

    Parameters
    ----------
    func : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    _converter = {
        ct.c_void_p: lambda x: x.ctypes.data,
        ct.c_int: lambda x: x,
        ct.POINTER(ct.c_int): lambda x: x.ctypes.data_as(ct.POINTER(ct.c_int)),
        ct.POINTER(ct.c_double): lambda x: x.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.POINTER(ct.c_float): lambda x: x.ctypes.data_as(ct.POINTER(ct.c_float)),
    }

    _func = func.ctypes
    print(_func.argtypes)
    _args = [_converter[j](i) for i, j in zip(args, _func.argtypes)]

    return _func(*_args)

def get_extension_path(lib_name):
    """
    Modified from rocket-fft
    """
    search_path = Path(__file__).parent#.parent
    ext_path = f"**/{lib_name}.*"
    matches = search_path.glob(ext_path)
    try:
        return str(next(matches))
    except StopIteration:
        return None
