import functools
import warnings
from collections import abc
from inspect import getfullargspec
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

IS_NPU_AVAILABLE=False
TORCH_VERSION = True
digit_version = True

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.autocast would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    # Note that when PyTorch >= 1.6.0, we still cast tensor types to fp16
    # manually, so the behavior may not be consistent with real amp.
    if IS_NPU_AVAILABLE:
        from torch.npu.amp import autocast
    else:
        from torch.cuda.amp import autocast
except ImportError:
    pass


def cast_tensor_type(inputs, src_type: torch.dtype, dst_type: torch.dtype):
    """Recursively convert Tensor in inputs from src_type to dst_type.
    Note:
        In v1.4.4 and later, ``cast_tersor_type`` will only convert the
        torch.Tensor which is consistent with ``src_type`` to the ``dst_type``.
        Before v1.4.4, it ignores the ``src_type`` argument, leading to some
        potential problems. For example,
        ``cast_tensor_type(inputs, torch.float, torch.half)`` will convert all
        tensors in inputs to ``torch.half`` including those originally in
        ``torch.Int`` or other types, which is not expected.
    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.
    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, torch.Tensor):
        # we need to ensure that the type of inputs to be casted are the same
        # as the argument `src_type`.
        return inputs.to(dst_type) if inputs.dtype == src_type else inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({  # type: ignore
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(  # type: ignore
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def auto_fp16(
        apply_to: Optional[Iterable] = None,
        out_fp32: bool = False,
        supported_types: tuple = (nn.Module, ),
) -> Callable:
    """Decorator to enable fp16 training automatically.
    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored. If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.
    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.
        supported_types (tuple): Classes can be decorated by ``auto_fp16``.
            `New in version 1.5.0.`
    Example:
        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp16
        >>>     @auto_fp16()
        >>>     def forward(self, x, y):
        >>>         pass
        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp16
        >>>     @auto_fp16(apply_to=('pred', ))
        >>>     def do_something(self, pred, others):
        >>>         pass
    """

    def auto_fp16_wrapper(old_func: Callable) -> Callable:

        @functools.wraps(old_func)
        def new_func(*args, **kwargs) -> Callable:
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], supported_types):
                raise TypeError('@auto_fp16 can only be used to decorate the '
                                f'method of those classes {supported_types}')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            with autocast(enabled=True):
                output = old_func(*new_args, **new_kwargs)

            # cast the results back to fp32 if necessary
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper
