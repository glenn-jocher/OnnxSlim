#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Sequence, Union

import numpy as np

from onnxslim.onnx_graphsurgeon.logger import G_LOGGER
from onnxslim.onnx_graphsurgeon.util import misc


class Tensor(object):
    """Abstract base class for tensors in a graph."""

    DYNAMIC = -1

    def __init__(self):
        """Initializes the Tensor class, serving as an abstract base class for tensors in a computational graph."""
        raise NotImplementedError("Tensor is an abstract class")

    def __setattr__(self, name, value):
        """Set an attribute, ensuring special handling for "inputs" and "outputs" properties."""
        if name in {"inputs", "outputs"}:
            try:
                attr = getattr(self, name)
                if value is attr:
                    # This can happen when using things like +=
                    # The __iadd__ is executed followed by an assignment
                    return

                attr.clear()
                attr.extend(value)
            except AttributeError:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def is_empty(self):
        """Determines if the tensor is used for an omitted optional input or output."""
        return self.name == ""

    def to_constant(
        self,
        values: np.ndarray,
        data_location: int = None,
        export_dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
    ):
        """Converts this tensor to a constant with specified values, data location, and data type."""
        self.__class__ = Constant
        self._values = values
        self.data_location = data_location
        self.export_dtype = export_dtype

        return self

    def to_variable(
        self, dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None, shape: Sequence[Union[int, str]] = None
    ):
        """Converts this tensor in-place to a Variable, updating all its consumers/producers accordingly."""

        if shape is None:
            shape = []
        variable_dtype = dtype if dtype is not None else self.export_dtype

        self.__class__ = Variable
        self.shape = shape
        self.dtype = variable_dtype

        return self

    def i(self, tensor_idx=0, producer_idx=0):
        """Returns the input tensor at the given index for a specified producer node in the graph."""
        return self.inputs[producer_idx].inputs[tensor_idx]

    def o(self, consumer_idx=0, tensor_idx=0):
        """Retrieve an output tensor from this tensor's specified output node and tensor indices."""
        return self.outputs[consumer_idx].outputs[tensor_idx]

    def __str__(self):
        """Returns a string with the tensor's type, name, shape, and data type representation."""
        return "{:} ({:}): (shape={:}, dtype={:})".format(type(self).__name__, self.name, self.shape, self.dtype)

    def __repr__(self):  # Hack to make logging output pretty.
        """Returns a string representation of the Tensor object for logging output."""
        return self.__str__()

    def __eq__(self, other):
        """Check if two tensors are equal based on their names."""
        return self.name == other.name


class Variable(Tensor):
    @staticmethod
    def empty():
        """Creates and returns an empty Variable tensor with an empty name."""
        return Variable(name="")

    def __init__(
        self,
        name: str,
        dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
        shape: Sequence[Union[int, str]] = None,
        type: str = "tensor_type",
    ):
        """Initialize a Variable tensor with name, data type, shape, and type attributes."""
        self.name = name
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=[])
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=[])
        self.dtype = dtype
        self.shape = misc.default_value(shape, None)
        self.type = type

    def to_constant(
        self,
        values: np.ndarray,
        export_dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
    ):
        """Modify a variable tensor in-place to convert it to a constant tensor with specified values and data type."""
        del self.dtype
        del self.shape

        return super().to_constant(values, export_dtype=export_dtype)

    def copy(self):
        """Creates a shallow copy of the tensor, excluding its input and output connections."""
        return Variable(self.name, self.dtype, self.shape)

    def __eq__(self, other):
        """Check if two Variable instances are equal by comparing names, inputs, outputs, dtype, shape, and type."""
        if not isinstance(other, Variable):
            return False

        name_match = self.name == other.name
        inputs_match = len(self.inputs) == len(other.inputs) and all(
            inp.name == other_inp.name for inp, other_inp in zip(self.inputs, other.inputs)
        )
        outputs_match = len(self.outputs) == len(other.outputs) and all(
            out.name == other_out.name for out, other_out in zip(self.outputs, other.outputs)
        )

        dtype_match = self.dtype == other.dtype
        shape_match = self.shape == other.shape
        type_match = self.type == other.type

        return name_match and inputs_match and outputs_match and dtype_match and shape_match and type_match


class LazyValues(object):
    """A special object that represents constant tensor values that should be lazily loaded."""

    def __init__(self, tensor):
        """Initialize the LazyValues object with the given ONNX tensor for lazy loading."""
        from onnxslim.onnx_graphsurgeon.importers.onnx_importer import (
            get_itemsize,
            get_onnx_tensor_dtype,
            get_onnx_tensor_shape,
        )

        self.tensor = tensor
        self.shape = get_onnx_tensor_shape(self.tensor)
        self.dtype = get_onnx_tensor_dtype(self.tensor)
        self.nbytes = misc.volume(self.shape) * get_itemsize(self.dtype)

    def load(self):
        """Load a numpy array from the tensor's underlying values."""
        import onnx
        import onnx.numpy_helper

        from onnxslim.onnx_graphsurgeon.importers.onnx_importer import (
            get_dtype_name,
            get_numpy_type,
        )

        if get_numpy_type(self.dtype) is None:
            G_LOGGER.warning(
                f"Datatype: {get_dtype_name(self.dtype)} could not be converted to a NumPy type.\n"
                f"Accessing the values of this constant tensor ({self.tensor.name}) will cause them to be casted to a supported data type. "
                f"This means that the weights will have a different type than the original model when they are exported again!\n"
                f"If this is not what you intended, please avoid accessing the values of this constant tensor."
            )

        return np.array(onnx.numpy_helper.to_array(self.tensor))

    def __str__(self):
        """Returns a formatted string representation of the LazyValues object indicating its shape and dtype."""
        return "LazyValues (shape={:}, dtype={:})".format(self.shape, self.dtype)

    def __repr__(self):  # Hack to make logging output pretty.
        """Returns an unambiguous string representation of the LazyValues object for logging purposes."""
        return self.__str__()

    def __eq__(self, other):
        """Check if two LazyValues instances have equal tensor data, shape, and dtype."""
        if not isinstance(other, LazyValues):
            return False

        tensor_match = self.tensor.raw_data == other.tensor.raw_data
        shape_match = self.shape == other.shape
        dtype_match = self.dtype == other.dtype

        return tensor_match and shape_match and dtype_match


class SparseValues(LazyValues):
    """A special object that represents constant tensor values that is sparse."""

    def load(self):
        """Loads a numpy array from the sparse tensor structure."""
        import onnx
        import onnx.numpy_helper

        supported_index_type = [onnx.TensorProto.INT64]
        if self.tensor.indices.data_type not in supported_index_type:
            G_LOGGER.critical(
                f"Unsupported index data type {self.tensor.indices.data_type} in {self.tensor.values.name}"
            )

        if self.tensor.values.data_type == onnx.TensorProto.FLOAT16:
            values_data = np.asarray(self.tensor.values.int32_data, dtype=np.uint16).view(np.float16)
        else:
            field_name = onnx.helper.tensor_dtype_to_field(self.tensor.values.data_type)
            values = getattr(self.tensor.values, field_name)
            dtype = onnx.helper.tensor_dtype_to_np_dtype(self.tensor.values.data_type)
            values_data = np.asarray(values, dtype)
        indices_data = self.tensor.indices.int64_data

        if len(self.tensor.indices.dims) == 1:
            values = np.zeros(np.prod(self.tensor.dims))
            # [NNZ] layout, in which case the i-th value must be the linearized-index of the i-th value.
            values[indices_data] = values_data
            values = values.reshape(self.tensor.dims)
        elif len(self.tensor.indices.dims) == 2:
            # [NNZ, rank] with the [i,j]-th value corresponding to the j-th index of the i-th value
            values = np.zeros(self.tensor.dims)
            indices_data = np.asarray(indices_data).reshape(self.tensor.indices.dims)

            for value_data, index_data in zip(values_data, indices_data):
                values[tuple(index_data)] = value_data
        else:
            G_LOGGER.critical(f"Unsupported index data dims {self.tensor.indices.dims} in {self.tensor.values.name}")

        return values

    def __str__(self):
        """Returns a formatted string representation of the SparseValues object indicating its shape and dtype."""
        return "SparseValues (shape={:}, dtype={:})".format(self.shape, self.dtype)


class Constant(Tensor):
    def __init__(
        self,
        name: str,
        values: Union[np.ndarray, LazyValues],
        data_location: int = None,
        export_dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
    ):
        """Initializes a Constant tensor with specific values, data location, and optional export data type."""
        self.name = name
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=[])
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=[])
        if (
            not isinstance(values, np.ndarray)
            and not isinstance(values, LazyValues)
            and not isinstance(values, SparseValues)
        ):
            G_LOGGER.critical(
                "Provided `values` argument is not a NumPy array, a LazyValues instance or a"
                "SparseValues instance. Please provide a NumPy array or LazyValues instance "
                "to construct a Constant. Note: Provided `values` parameter was: {:}".format(values)
            )
        self._values = values
        self.data_location = data_location
        self._export_dtype = export_dtype

    def to_variable(self, dtype: np.dtype = None, shape: Sequence[Union[int, str]] = None):
        """Convert instance values to an appropriate variable with specified dtype and shape."""
        if shape is None:
            shape = []
        del self._export_dtype
        del self._values

        if dtype is not None:
            return super().to_variable(dtype, shape)

        var_dtype = self.export_dtype

        return super().to_variable(var_dtype, shape)

    def copy(self):
        """Creates a shallow copy of the Constant tensor, excluding inputs and outputs."""
        return Constant(self.name, self._values, export_dtype=self.export_dtype)

    @property
    def values(self):
        """Retrieve tensor values, lazily loading them if accessed for the first time."""
        if isinstance(self._values, LazyValues):
            self._values = self._values.load()
        return self._values

    @values.setter
    def values(self, values: Union[np.ndarray, LazyValues]):
        """Return tensor values, loading them if being accessed for the first time."""
        self._values = values

    @property
    def shape(self):
        """Retrieve the shape of the tensor values."""
        return self._values.shape

    @property
    def dtype(self):
        """Retrieve the data type (dtype) of this tensor's values."""
        return self._values.dtype

    @property
    def export_dtype(self):
        """Returns the export data type (export_dtype) for this Constant tensor."""
        return self._export_dtype if self._export_dtype is not None else self.dtype

    @export_dtype.setter
    def export_dtype(self, export_dtype):
        """Get the tensor's export data type, defaulting to its current dtype if unspecified."""
        self._export_dtype = export_dtype

    def __repr__(self):  # Hack to make logging output pretty.
        """Returns a string representation of the Constant object, including its values, for logging purposes."""
        ret = self.__str__()
        ret += "\n{:}".format(self._values)
        return ret

    def __eq__(self, other):
        """Compare two `Constant` tensors for equality based on their values."""
        if not isinstance(other, Constant):
            return False

        return (
            self._values == other._values
            if isinstance(self._values, LazyValues) and isinstance(other._values, LazyValues)
            else np.array_equal(self.values, other.values)
        )
