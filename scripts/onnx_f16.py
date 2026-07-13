"""Halve ONNX file size by storing float32 initializers as float16 + a Cast node.

All computation stays in fp32 (ONNX Runtime constant-folds Cast(initializer) at session load),
so outputs differ from fp32 only by fp16 rounding of the weights. This avoids
onnxconverter_common.float16, which fails on dynamo-exported graphs containing Cast nodes.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_ELEMENTS = 1024  # leave tiny tensors (layernorm params, biases) in fp32


def convert_initializers_to_f16(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    new_inits = []
    casts = []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.size >= MIN_ELEMENTS:
            f16_name = f"{init.name}__f16"
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), f16_name))
            casts.append(helper.make_node("Cast", [f16_name], [init.name], to=TensorProto.FLOAT))
        else:
            new_inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(new_inits)
    # Cast nodes must precede any consumer; prepend them.
    old_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + old_nodes)
    return model


def save_f16(src_path: str, dst_path: str) -> None:
    model = onnx.load(src_path)
    onnx.save(convert_initializers_to_f16(model), dst_path)
