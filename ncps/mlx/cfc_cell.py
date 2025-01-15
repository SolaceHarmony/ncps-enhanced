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


import numpy as np
import coremltools as ct
from typing import Optional, Union


# LeCun improved tanh activation
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
def lecun_tanh(x):
    return 1.7159 * np.tanh(0.666 * x)


class CfCCell(ct.models.MLModel):
    def __init__(
        self,
        units,
        input_sparsity=None,
        recurrent_sparsity=None,
        mode="default",
        activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.1,
        **kwargs,
    ):
        """A `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps.
            To get a full RNN that can process sequences,
            see `ncps.mlx.CfC` or wrap the cell with a `ct.models.MLModel`.

        :param units: Number of hidden units
        :param input_sparsity:
        :param recurrent_sparsity:
        :param mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate).
        :param activation: Activation function used in the backbone layers
        :param backbone_units: Number of hidden units in the backbone layer (default 128)
        :param backbone_layers: Number of backbone layers (default 1)
        :param backbone_dropout: Dropout rate in the backbone layers (default 0)
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.units = units
        self.sparsity_mask = None
        if input_sparsity is not None or recurrent_sparsity is not None:
            # No backbone is allowed
            if backbone_units > 0:
                raise ValueError(
                    "If sparsity of a Cfc cell is set, then no backbone is allowed"
                )
            # Both need to be set
            if input_sparsity is None or recurrent_sparsity is None:
                raise ValueError(
                    "If sparsity of a Cfc cell is set, then both input and recurrent sparsity needs to be defined"
                )
            self.sparsity_mask = np.concatenate([input_sparsity, recurrent_sparsity], axis=0)

        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                "Unknown mode '{}', valid options are {}".format(
                    mode, str(allowed_modes)
                )
            )
        self.mode = mode
        self.backbone_fn = None
        if activation == "lecun_tanh":
            activation = lecun_tanh
        self._activation = activation
        self._backbone_units = backbone_units
        self._backbone_layers = backbone_layers
        self._backbone_dropout = backbone_dropout
        self._cfc_layers = []

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple) or isinstance(
            input_shape[0], np.ndarray
        ):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        backbone_layers = []
        for i in range(self._backbone_layers):
            backbone_layers.append(
                ct.models.MLModel(
                    self._backbone_units, self._activation, name=f"backbone{i}"
                )
            )
            backbone_layers.append(ct.models.MLModel(self._backbone_dropout))

        self.backbone_fn = ct.models.MLModel(backbone_layers)
        cat_shape = int(
            self.state_size + input_dim
            if self._backbone_layers == 0
            else self._backbone_units
        )
        if self.mode == "pure":
            self.ff1_kernel = self.add_weight(
                shape=(cat_shape, self.state_size),
                initializer="glorot_uniform",
                name="ff1_weight",
            )
            self.ff1_bias = self.add_weight(
                shape=(self.state_size,),
                initializer="zeros",
                name="ff1_bias",
            )
            self.w_tau = self.add_weight(
                shape=(1, self.state_size),
                initializer=np.zeros,
                name="w_tau",
            )
            self.A = self.add_weight(
                shape=(1, self.state_size),
                initializer=np.ones,
                name="A",
            )
        else:
            self.ff1_kernel = self.add_weight(
                shape=(cat_shape, self.state_size),
                initializer="glorot_uniform",
                name="ff1_weight",
            )
            self.ff1_bias = self.add_weight(
                shape=(self.state_size,),
                initializer="zeros",
                name="ff1_bias",
            )
            self.ff2_kernel = self.add_weight(
                shape=(cat_shape, self.state_size),
                initializer="glorot_uniform",
                name="ff2_weight",
            )
            self.ff2_bias = self.add_weight(
                shape=(self.state_size,),
                initializer="zeros",
                name="ff2_bias",
            )

            self.time_a = ct.models.MLModel(self.state_size, name="time_a")
            self.time_b = ct.models.MLModel(self.state_size, name="time_b")
        self.built = True

    def call(self, inputs, states, **kwargs):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = np.reshape(t, [-1, 1])
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            t = 1.0
        x = np.concatenate([inputs, states[0]], axis=-1)
        x = self.backbone_fn(x)
        if self.sparsity_mask is not None:
            ff1_kernel = self.ff1_kernel * self.sparsity_mask
            ff1 = np.dot(x, ff1_kernel) + self.ff1_bias
        else:
            ff1 = np.dot(x, self.ff1_kernel) + self.ff1_bias
        if self.mode == "pure":
            # Solution
            new_hidden = (
                -self.A
                * np.exp(-t * (np.abs(self.w_tau) + np.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            if self.sparsity_mask is not None:
                ff2_kernel = self.ff2_kernel * self.sparsity_mask
                ff2 = np.dot(x, ff2_kernel) + self.ff2_bias
            else:
                ff2 = np.dot(x, self.ff2_kernel) + self.ff2_bias
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = 1 / (1 + np.exp(-t_a * t + t_b))
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, [new_hidden]
