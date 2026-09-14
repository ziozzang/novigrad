"""Experimental Metal batch inference preserving sparse edge order.

Requires the optional MLX dependencies on Apple silicon. Use Engine for learning
and low-latency single observations. This module is not an MLX-C binding.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import Engine
from ._mlx_ops import build


class MlxEngine:
    def __init__(self, path):
        if not mx.metal.is_available():
            raise RuntimeError('MlxEngine requires an available Apple Metal GPU')
        path = Path(path)
        # Rust is the checkpoint validator, including graph indices and schema.
        core = Engine.load(path)
        self._ids = core.input_ids
        self._config = core.config
        self._output_actions = core.output_actions
        self._output_gains = core.output_gains
        with mx.stream(mx.gpu):
            forward, ids, setup = build(path, include_dense=False)
            if ids.tolist() != self._ids:
                raise ValueError('checkpoint port order changed during loading')
            self._forward = mx.compile(lambda inputs: forward(inputs, sparse=True))
        self.setup = setup

    @classmethod
    def load(cls, path):
        return cls(path)

    @property
    def input_ids(self):
        return list(self._ids)

    @property
    def config(self):
        return dict(self._config)

    @property
    def output_actions(self):
        return list(self._output_actions)

    @property
    def output_gains(self):
        return list(self._output_gains)

    def infer(self, rates):
        return self.infer_batch([rates])[0]

    def infer_batch(self, rows):
        inputs = np.asarray(rows, dtype=np.float32)
        if inputs.ndim != 2 or len(inputs) == 0 or inputs.shape[1] != len(self._ids) or not np.isfinite(inputs).all() or np.any(inputs < 0):
            raise ValueError('expected nonempty finite nonnegative rates [batch,input_ports]')
        with mx.stream(mx.gpu):
            output = self._forward(mx.array(inputs))
            mx.eval(output)
            return output.tolist()
