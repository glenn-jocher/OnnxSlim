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

import copy
from typing import List, Sequence

from onnxslim.onnx_graphsurgeon.ir.graph import Graph
from onnxslim.onnx_graphsurgeon.ir.node import Node
from onnxslim.onnx_graphsurgeon.ir.tensor import Tensor, Variable
from onnxslim.onnx_graphsurgeon.logger import G_LOGGER
from onnxslim.onnx_graphsurgeon.util import misc


class Function(Graph):
    """
    Represents a local function, which is a default implementation of a Custom Op. This default implementation is
    represented as a Graph of other Ops.

    Functions are used in a model by creating a Node with the same name and domain as the function. This can be done
    using the __call__() method of a Function, which creates this new node and appends it to a Graph. A Function is not
    a subgraph of a Graph, and its Nodes, Tensors, and subgraphs are entirely separate from the main Graph.

    Functions can be composed of other functions, but cyclical or recursive definitions are not allowed in ONNX.
    """

    DEFAULT_DOMAIN = "onnx_graphsurgeon"

    def __init__(
        self,
        name: str,
        domain: str = None,
        nodes: Sequence[Node] = None,
        inputs: Sequence[Tensor] = None,
        outputs: Sequence[Tensor] = None,
        doc_string: str = None,
        opset: int = None,
        import_domains: "Sequence[onnx.OperatorSetIdProto]" = None,
        functions: "Sequence[Function]" = None,
        attrs: dict = None,
    ):
        """Initializes the Function instance with specified name, domain, nodes, and attributes for ONNX models."""
        self.domain = misc.default_value(domain, Function.DEFAULT_DOMAIN)
        self.attrs = misc.default_value(attrs, {})

        super().__init__(
            nodes,
            inputs,
            outputs,
            name=name,
            doc_string=doc_string,
            opset=opset,
            import_domains=import_domains,
            functions=functions,
        )

        # Properties of Graph that Function doesn't have.
        del self.producer_name
        del self.producer_version

    @property
    def unique_id(self):
        """Returns a tuple uniquely identifying this function."""
        return (self.domain, self.name)

    def cleanup(
        self,
        remove_unused_node_outputs=False,
        recurse_subgraphs=True,
        remove_unused_graph_inputs=False,
        recurse_functions=False,
    ):
        """Cleans up the function graph, removing unused nodes and tensors. See http://www.apache.org/licenses/LICENSE-2.0"""
        if recurse_functions:
            G_LOGGER.warning(
                "Function.cleanup() called with recurse_functions=True, meaning that other functions will also be cleaned up."
            )
        return super().cleanup(
            remove_unused_node_outputs=remove_unused_node_outputs,
            recurse_subgraphs=recurse_subgraphs,
            remove_unused_graph_inputs=remove_unused_graph_inputs,
            recurse_functions=recurse_functions,
        )

    def fold_constants(self, recurse_functions=False, **kwargs):
        """Fold constants in the Function's graph; optionally recurse into nested functions."""
        if recurse_functions:
            G_LOGGER.warning(
                "Function.fold_constants() called with recurse_functions=True, meaning that other functions will also be const-folded."
            )
        return super().fold_constants(recurse_functions=recurse_functions, **kwargs)

    def toposort(
        self,
        recurse_subgraphs=True,
        recurse_functions=False,
        mode="nodes",
    ):
        """Perform topological sorting of function nodes, defaulting to not sorting other functions."""
        if recurse_functions:
            G_LOGGER.warning(
                "Function.toposort() called with recurse_functions=True, meaning that other functions will be sorted."
            )
        return super().toposort(
            recurse_subgraphs=recurse_subgraphs,
            recurse_functions=recurse_functions,
            mode=mode,
        )

    def __call__(self, graph, inputs=None, outputs=None, *args, **kwargs) -> List[Tensor]:
        """Instantiates this Function as a Node within a graph, processing inputs and outputs accordingly."""
        if inputs is not None and len(inputs) != len(self.inputs):
            msg_template = "Function {} expects {} inputs, but was called with {} inputs."
            G_LOGGER.warning(msg_template.format(self.name, len(self.inputs), len(inputs)))

        new_output_indices = []
        if outputs is None:
            # Graph.layer() will create Tensors and make sure the names do not conflict.
            outputs = [out.name for out in self.outputs]
            new_output_indices = list(range(len(outputs)))
        elif len(outputs) != len(self.outputs):
            msg_template = "Function {} expects {} outputs, but was called with {} outputs."
            G_LOGGER.warning(msg_template.format(self.name, len(self.outputs), len(outputs)))
        else:
            new_output_indices = [i for i in range(len(outputs)) if not isinstance(outputs[i], Tensor)]

        attrs = kwargs.get("attrs", None)
        if attrs is not None:
            for attr_name, default_val in self.attrs.items():
                if default_val is None and attr_name not in attrs:
                    msg_template = "Function {} called without required attribute: {}"
                    G_LOGGER.warning(msg_template.format(self.name, attr_name))

        inputs = misc.default_value(inputs, [])
        outputs = misc.default_value(outputs, [])
        outputs = graph.layer(
            *args,
            **kwargs,
            op=self.name,
            domain=self.domain,
            inputs=inputs,
            outputs=outputs,
        )

        # For newly created output tensors, set their shape and dtype to match the Function definition.
        for i in new_output_indices:
            outputs[i].dtype = self.outputs[i].dtype
            outputs[i].shape = self.outputs[i].shape

        return outputs

    def copy(self):
        """Creates a deep copy of the function, including nodes and tensors but not weights or non-Graph attributes."""

        local_tensor_copies = {n: t.copy() for n, t in self.tensors().items()}

        def get_tensor(name):
            """Retrieve a tensor by name from a deep-copied dictionary of tensors."""
            return local_tensor_copies[name] if name else Variable.empty()

        # Next, copy nodes, and update inputs/outputs
        new_nodes = []
        for node in self.nodes:
            new_node = node.copy(
                inputs=[get_tensor(inp.name) for inp in node.inputs],
                outputs=[get_tensor(out.name) for out in node.outputs],
                tensor_map=local_tensor_copies,
            )
            new_nodes.append(new_node)
        new_func_inputs = [get_tensor(inp.name) for inp in self.inputs]
        new_func_outputs = [get_tensor(out.name) for out in self.outputs]

        new_attrs = {name: copy.copy(val) for name, val in self.attrs.items()}

        return Function(
            self.name,
            self.domain,
            nodes=new_nodes,
            inputs=new_func_inputs,
            outputs=new_func_outputs,
            doc_string=self.doc_string,
            opset=self.opset,
            import_domains=self.import_domains,
            functions=self.functions,
            attrs=new_attrs,
        )

    def __eq__(self, other: "Function"):
        """Determine equality based on function attributes and unique identifiers."""

        def sequences_equal(seq1, seq2):
            """Checks if two sequences are equal in length and elements."""
            return len(seq1) == len(seq2) and all(elem1 == elem2 for elem1, elem2 in zip(seq1, seq2))

        return (
            self.unique_id == other.unique_id
            and self.opset == other.opset
            and self.import_domains == other.import_domains
            and sequences_equal(self.inputs, other.inputs)
            and sequences_equal(self.outputs, other.outputs)
            and sequences_equal(self.nodes, other.nodes)
        )

    def __str__(self):
        """Returns a string representation of the function with its name, domain, opset, inputs, nodes, and outputs."""
        nodes_str = "\n".join([str(node) for node in self.nodes])
        out = f"Function {self.name}, Domain {self.domain}, Opset {self.opset}"
        out += f"\nInputs: {self.inputs}"
        out += f"\nNodes: {nodes_str}"
        out += f"\nOutputs: {self.outputs}"
        return out
