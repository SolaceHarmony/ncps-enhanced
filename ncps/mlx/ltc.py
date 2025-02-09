# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ncps
from . import LTCCell, MixedMemoryRNN
# import tensorflow as tf
from typing import Union, Optional, Dict, Any

@ncps.mini_keras.utils.register_keras_serializable(package="ncps", name="LTC")
class LTC(ncps.mini_keras.layers.RNN):
    """
    Applies a Liquid time-constant (LTC) RNN to an input sequence.

    Examples:
        >>> from ncps.tf import LTC
        >>>
        >>> rnn = LTC(50)
        >>> x = tf.random.uniform((2, 10, 20))  # (B,L,C)
        >>> y = rnn(x)

    Note:
        For creating a wired Neural circuit policy (NCP) you can pass a `ncps.wirings.NCP` object instead of the number of units

    Examples:
        >>> from ncps.tf import LTC
        >>> from ncps.wirings import NCP
        >>>
        >>> wiring = NCP(10, 10, 8, 6, 6, 4, 4)
        >>> rnn = LTC(wiring)
        >>> x = tf.random.uniform((2, 10, 20))  # (B,L,C)
        >>> y = rnn(x)
    """

    def __init__(
    self,
    units: Union[int, ncps.wirings.Wiring],
    input_mapping: str = "affine",
    output_mapping: str = "affine",
    ode_unfolds: int = 6,
    epsilon: float = 1.0,
    initialization_ranges: Optional[Dict[str, Any]] = None,
    return_sequences: bool = False,
    return_state: bool = False,
    go_backwards: bool = False,
    stateful: bool = False,
    unroll: bool = False,
    time_major: bool = False,
    mixed_memory: bool = False,
    **kwargs
    ) -> None:
        """
        Initialize the LTC RNN.

        Args:
            units: Wiring (ncps.wirings.Wiring instance) or integer representing the number of (fully-connected) hidden units
            mixed_memory: Whether to augment the RNN by a memory-cell to help learn long-term dependencies in the data
            input_mapping: Mapping applied to the sensory neurons. Possible values None, "linear", "affine" (default "affine")
            output_mapping: Mapping applied to the motor neurons. Possible values None, "linear", "affine" (default "affine")
            ode_unfolds: Number of ODE-solver steps per time-step (default 6)
            epsilon: Auxillary value to avoid dividing by 0 (default 1e-8)
            initialization_ranges: A dictionary for overwriting the range of the uniform weight initialization (default None)
            return_sequences: Whether to return the full sequence or just the last output (default False)
            return_state: Whether to return just the output of the RNN or a tuple (output, last_hidden_state) (default False)
            go_backwards: If True, the input sequence will be process from back to the front (default False)
            stateful: Whether to remember the last hidden state of the previous inference/training batch and use it as initial state for the next inference/training batch (default False)
            unroll: Whether to unroll the graph, i.e., may increase speed at the cost of more memory (default False)
            time_major: Whether the time or batch dimension is the first (0-th) dimension (default False)
            kwargs: Additional arguments.
        """

        wiring = ncps.wirings.FullyConnected(units) if isinstance(units, int) else units


        cell = LTCCell(
            wiring=wiring,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            initialization_ranges=initialization_ranges,
            **kwargs,
        )
        if mixed_memory:
            cell = MixedMemoryRNN(cell)
        import mlx.core as mx

        super(LTC, self).__init__(
            cell,
            return_sequences,
            return_state,
            go_backwards,
            stateful,
            unroll,
            time_major,
            **kwargs,
        )
