from math import log, pi
from dataclasses import fields
import importlib
import torch
from torch.nn.functional import interpolate
import functools
import weakref
import sys
import contextlib
from tqdm.contrib import DummyTqdmFile
import tqdm


def rescale_tensor(tensor, width, height):
    assert tensor.dim() == 3 and tensor.shape[0] == 3, "tensor must be 3xHxW"
    tensor = tensor.unsqueeze(0)
    tensor = interpolate(tensor, (height, width), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def tensor(*v, dtype=torch.float32, device=None):
    return torch.tensor(v, dtype=dtype, device=device)


def ishomo(points):
    if points.dim() == 1:
        return points.shape[0] == 4
    elif points.dim() == 2:
        return points.shape[1] == 4
    return False


def homo(points):
    if points.dim() == 1:
        if points.shape[0] == 4:
            return points
        elif points.shape[0] == 3:
            return torch.cat(
                [points, torch.tensor([1], dtype=points.dtype, device=points.device)]
            )
    elif points.dim() == 2:
        if points.shape[1] == 4:
            return points
        if points.shape[1] == 3:
            return torch.cat(
                [
                    points,
                    torch.ones(
                        (points.shape[0], 1), dtype=points.dtype, device=points.device
                    ),
                ]
            )

    raise ValueError(f"points does not match with any valid shapes. {points.shape}")


def unhomo(points):
    if points.dim() == 1:
        if points.shape[0] == 3:
            return points
        elif points.shape[0] == 4:
            return points[:3] / (points[3])
    elif points.dim() == 2:
        if points.shape[1] == 3:
            return points
        if points.shape[1] == 4:
            return points[:, 3] / (points[:, 3])

    raise ValueError(f"points does not match with any valid shapes. {points.shape}")


def inverse_sigmoid(x):
    if isinstance(x, float):
        return -log(1 / x - 1)
    return torch.log(x / (1 - x))


def accumulate_tensor(tensor, index, value):
    assert (
        tensor.shape[:-1] == value.shape[:-1]
    ), "tensor and value must have the same shape, except for the last dimension"
    assert (
        index.shape == value.shape[-1:]
    ), "index must be a 1D tensor with the same length as the last dimension of value"

    accumulated_tensor = torch.zeros_like(tensor)
    counter = torch.zeros_like(tensor)

    accumulated_tensor.index_add_(-1, index, value)
    counter.index_add_(-1, index, torch.ones_like(value))

    return tensor + accumulated_tensor, counter


def attach_direction_prompt(prompt, elevs, azims):
    if type(elevs) == float:
        elevs = [elevs]
    if type(azims) == float:
        azims = [azims]

    output_prompts = []

    for elev, azim in zip(elevs, azims):
        direction_prompt = ""
        # elev 60~: top view
        # azim -45~45: front view
        # azim 45~135: right view
        # azim 135~225: back view
        # azim 225~315: left view
        if elev > 60:
            direction_prompt = "top view"
        elif azim < 45 or azim > 315:
            direction_prompt = "front view"
        elif azim < 135:
            direction_prompt = "right view"
        elif azim < 225:
            direction_prompt = "back view"
        else:
            direction_prompt = "left view"
        output_prompts.append(f"{prompt}, {direction_prompt}")

    return output_prompts


def attach_elevation_prompt(prompt, elevs):
    if type(elevs) == float:
        elevs = [elevs]

    output_prompts = []

    for elev in elevs:
        direction_prompt = ""
        # elev -90~-80: "viewed from directly above"
        # elev -80~-45: "high-angle downward view"
        # elev -45~-10: "elevated downward view"
        # elev -10~10: "eye-level view"
        # elev 10~45: "low-angle upward view"
        # elev 45~80: "steep upward view"
        # elev 80~: "viewed from directly below"
        if elev < -80:
            direction_prompt = "viewed from directly above"
        elif elev < -45:
            direction_prompt = "high-angle downward view"
        elif elev < -10:
            direction_prompt = "elevated downward view"
        elif elev < 10:
            direction_prompt = "eye-level view"
        elif elev < 45:
            direction_prompt = "low-angle upward view"
        elif elev < 80:
            direction_prompt = "steep upward view"
        else:
            direction_prompt = "viewed from directly below"
        output_prompts.append(f"{prompt}, {direction_prompt}")

    return output_prompts


def ignore_kwargs(cls):
    original_init = cls.__init__

    def init(self, *args, **kwargs):
        expected_fields = {field.name for field in fields(cls)}
        expected_kwargs = {
            key: value for key, value in kwargs.items() if key in expected_fields
        }
        original_init(self, *args, **expected_kwargs)

    cls.__init__ = init
    return cls


def get_class_filename(cls):
    """
    Get the filename of the module containing the given class.

    Args:
    cls (type): The class for which to find the filename.

    Returns:
    str: The filename of the module containing the class.
    """
    # Get the module name where the class is defined
    module_name = cls.__module__

    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Retrieve the filename of the module
    filename = module.__file__

    return filename


def weak_lru(maxsize=128, typed=False):
    'LRU Cache decorator that keeps a weak reference to "self"'

    def wrapper(func):

        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


@contextlib.contextmanager
def redirect_stdout_to_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


ORIG_STDOUT = sys.stdout
ORIG_STDERR = sys.stderr


def redirected_tqdm(*args, **kwargs):
    return tqdm.tqdm(*args, file=ORIG_STDOUT, **kwargs)


def redirected_trange(*args, **kwargs):
    return tqdm.trange(*args, file=ORIG_STDOUT, **kwargs)


if __name__ == "__main__":

    def dummy():
        print("dummy")

    def dummy_loop():
        for j in redirected_trange(3, desc="inner", leave=False, position=1):
            print(i, j)
            sleep(0.5)
            dummy()
            sleep(0.5)

    from time import sleep

    with redirect_stdout_to_tqdm():
        for i in redirected_trange(2, desc="outer", leave=False, position=0):
            dummy_loop()
