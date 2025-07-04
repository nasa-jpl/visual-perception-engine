from functools import wraps

import torch


def to_shapes(data_dict: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    """Convert a dictionary of tensors to a dictionary of their shapes."""
    return {key: tensor.shape for key, tensor in data_dict.items()}


def compatible_shapes(shape1: list[int | None], shape2: list[int | None]) -> bool:
    """Check if the two shapes are compatible, i.e. shape1 can fit into shape2. None values are treated as wildcards."""
    if len(shape1) != len(shape2):
        return False

    # None in any shape is treated as a wildcard and as such it can be any value
    for s1, s2 in zip(shape1, shape2):
        if (s1 is None and s2 is not None) or (
            s2 is not None and s1 != s2
        ):  # this can be simplified but this way it is more readable
            return False

    return True


def is_io_compatible(signature1: dict[str, tuple], signature2: dict[str, tuple]) -> bool:
    """Check if the signature1 can be used as an input for signature2."""
    matching_keys = set(signature2.keys()).issubset(set(signature1.keys()))
    if not matching_keys:
        return False
    matching_shapes = all([compatible_shapes(signature1[key], signature2[key]) for key in signature2.keys()])
    return matching_keys and matching_shapes


def assert_correct_io_shapes(func):
    """Decorator that asserts that the input and output shapes of the function are correct.
    Can be applied to models implementing ModelInterfaceBase or transforms implementing AbstractTransform."""

    @wraps(func)
    def wrapper(self, annotated_input: dict[str, torch.Tensor]):
        assert is_io_compatible(to_shapes(annotated_input), self.input_signature), (
            f"Input signature mismatch. Expected {self.input_signature}, got {to_shapes(annotated_input)}"
        )
        out = func(self, annotated_input)
        assert is_io_compatible(to_shapes(out), self.output_signature), (
            f"Output signature mismatch. Expected {self.output_signature}, got {to_shapes(out)}"
        )
        return out

    return wrapper


def assert_correct_types(func):
    """Decorator that asserts that the input and output types of the function are correct.
    Can be applied to transforms implementing AbstractTransform."""

    @wraps(func)
    def wrapper(self, annotated_input: dict[str, torch.Tensor]):
        assert all([annotated_input[key].dtype == self.input_type for key in annotated_input.keys()]), (
            f"Input type mismatch. Expected {self.input_type}, got {[annotated_input[key].dtype for key in annotated_input.keys()]}"
        )
        out = func(self, annotated_input)
        assert all([out[key].dtype == self.output_type for key in out.keys()]), (
            f"Output type mismatch. Expected {self.output_type}, got {[out[key].dtype for key in out.keys()]}"
        )
        return out

    return wrapper
