from typing import Any

import torch
from torch2trt import TRTModule, torch_dtype_from_trt, torch_device_from_trt

from model_architectures.interfaces import ModelInterfaceBase


class CustomTRTModule(TRTModule, ModelInterfaceBase):
    def __init__(self, trt_model: TRTModule, base_model: ModelInterfaceBase) -> None:
        super(CustomTRTModule, self).__init__(
            trt_model.engine,
            trt_model.input_names,
            trt_model.output_names,
            trt_model.input_flattener,
            trt_model.output_flattener,
        )
        self.base_model = base_model

        self.output_memory_slot = {}  # this will be initialized in the first forward pass

    def _forward_post_10(self, *inputs):
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)

        # set shapes
        for i, input_name in enumerate(self.input_names):
            if input_name not in self._name_to_binding:
                # it is possible that TensorRT removed inputs that are not used for computations
                continue
            shape = tuple(inputs[i].shape)
            data_ptr = inputs[i].contiguous().data_ptr()
            self.context.set_tensor_address(input_name, data_ptr)
            self.context.set_input_shape(input_name, shape)

        # execute
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            # create reusable output tensor
            if output_name not in self.output_memory_slot:
                dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
                shape = tuple(self.context.get_tensor_shape(output_name))
                device = torch_device_from_trt(self.engine.get_tensor_location(output_name))
                output = torch.empty(size=shape, dtype=dtype, device=device)
                self.output_memory_slot[output_name] = output
            else:
                output = self.output_memory_slot[output_name]
            outputs[i] = output
            self.context.set_tensor_address(output_name, output.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]

        return outputs

    def deannotate_input(self, x: dict[str, torch.Tensor]) -> Any:
        return self.base_model.deannotate_input(x)

    def annotate_output(self, x: Any) -> dict[str, torch.Tensor]:
        return self.base_model.annotate_output(x)

    @property
    def input_signature(self) -> dict[str, tuple[int]]:
        return self.base_model.input_signature

    @property
    def output_signature(self) -> dict[str, tuple[int]]:
        return self.base_model.output_signature

    @property
    def is_model_head(self) -> bool:
        return self.base_model.is_model_head
