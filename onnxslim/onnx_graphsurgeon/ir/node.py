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

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

from onnxslim.onnx_graphsurgeon.ir.tensor import Tensor
from onnxslim.onnx_graphsurgeon.logger import G_LOGGER
from onnxslim.onnx_graphsurgeon.util import misc


class Node(object):
    @dataclass
    class AttributeRef:
        """
        An AttributeRef is an attribute value which references an attribute in the parent function. A node's attribute
        can only be an AttributeRef if the node lives inside a Function.

        Args:
            name (str): The name of the referenced attribute in the parent Function.
            type (type): The attribute's type.
        """

        name: str
        type: type

    def __init__(
        self,
        op: str,
        name: str = None,
        attrs: Dict[str, object] = None,
        inputs: List["Tensor"] = None,
        outputs: List["Tensor"] = None,
        domain: str = None,
    ):
        """Initializes a Node representing an operation in a graph with attributes, inputs, and outputs."""
        self.op = op
        self.name = misc.default_value(name, "")
        self.attrs = misc.default_value(attrs, OrderedDict())
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=misc.default_value(inputs, []))
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=misc.default_value(outputs, []))
        self.domain = domain

    def i(self, tensor_idx=0, producer_idx=0):
        """Fetches a producer node of this node's input tensor at the specified indices, defaults to (0, 0)."""
        return self.inputs[tensor_idx].inputs[producer_idx]

    def o(self, consumer_idx=0, tensor_idx=0):
        """Retrieve a consumer node of a specified output tensor from this node."""
        return self.outputs[tensor_idx].outputs[consumer_idx]

    def subgraphs(self, recursive=False):
        """Iterate over all subgraphs contained in this node, with optional recursive exploration."""
        from onnxslim.onnx_graphsurgeon.ir.graph import Graph

        visit_queue = [self]

        # This prevents infinite recursion in the (illegal) case of cyclical graphs.
        visited = set()

        while visit_queue:
            node = visit_queue.pop()
            for attr in node.attrs.values():
                if isinstance(attr, Graph) and id(attr) not in visited:
                    visited.add(id(attr))
                    if recursive:
                        visit_queue.extend(attr.nodes)
                    yield attr

    def __setattr__(self, name, value):
        """Sets the attribute 'name' to 'value', with special handling for 'inputs' and 'outputs' attributes."""
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

    def copy(
        self,
        inputs: List["Tensor"] = None,
        outputs: List["Tensor"] = None,
        tensor_map=None,
    ):
        """Shallowly copies the node with the option to override inputs, outputs, and tensor mappings."""
        from onnxslim.onnx_graphsurgeon.ir.graph import Graph

        new_attrs = OrderedDict()
        for name, attr in self.attrs.items():
            new_attrs[name] = attr.copy(tensor_map) if isinstance(attr, Graph) else attr
        return Node(
            self.op,
            self.name,
            new_attrs,
            inputs=inputs,
            outputs=outputs,
            domain=self.domain,
        )

    def __str__(self):
        """Returns a string representation of the node, displaying its name, operation, inputs, outputs, and
        attributes.
        """
        ret = "{:} ({:})".format(self.name, self.op)

        def add_io(name, io):
            """Add the input or output operations and their names to the string representation of the object."""
            nonlocal ret
            ret += "\n\t{:}: [".format(name)
            for elem in io:
                ret += "\n\t\t{:}".format(elem)
            ret += "\n\t]"

        add_io("Inputs", self.inputs)
        add_io("Outputs", self.outputs)

        if self.attrs:
            ret += "\nAttributes: {:}".format(self.attrs)

        if self.domain:
            ret += "\nDomain: {:}".format(self.domain)

        return ret

    def __repr__(self):
        """Return the string representation of the Node object."""
        return self.__str__()

    def __eq__(self, other):
        """Check whether two nodes are equal by comparing the name, attributes, op, inputs, and outputs."""
        G_LOGGER.verbose("Comparing node: {:} with {:}".format(self.name, other.name))
        attrs_match = self.name == other.name and self.op == other.op and self.attrs == other.attrs
        if not attrs_match:
            return False

        inputs_match = misc.sequences_equal(self.inputs, other.inputs)
        if not inputs_match:
            return False

        outputs_match = misc.sequences_equal(self.outputs, other.outputs)
        return self.domain == other.domain if outputs_match else False
