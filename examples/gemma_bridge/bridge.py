"""Frozen semantic encoder -> nonnegative neural ports -> reward policy."""
from pathlib import Path
import numpy as np
from novigrad import Engine


class SemanticPorts:
    def __init__(self, input_count=319, dimensions=128):
        if dimensions < 1 or dimensions * 2 > input_count:
            raise ValueError('signed embedding needs two ports per dimension')
        self.input_count, self.dimensions = input_count, dimensions

    def encode(self, embeddings):
        x = np.asarray(embeddings, np.float32)
        if x.ndim != 2 or x.shape[1] < self.dimensions or not np.isfinite(x).all():
            raise ValueError('expected finite embedding matrix of sufficient width')
        x = x[:, :self.dimensions].copy()
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        rates = np.zeros((len(x), self.input_count), np.float32)
        rates[:, :self.dimensions] = np.maximum(x, 0)
        rates[:, self.dimensions:2*self.dimensions] = np.maximum(-x, 0)
        return rates


class NoviPolicy:
    def __init__(self, root, learning_rate=.03, logit_gain=6.):
        self.engine = Engine.from_edges(Path(root)/'data/pn_kc.tsv', Path(root)/'data/kc_mbon.tsv',
            actions=4, learning_rate=learning_rate, logit_gain=logit_gain, active_fraction=.2,
            homeostasis=False, readout='opponent')

    def predict(self, x):
        return np.asarray(self.engine.infer_batch(x.tolist()), np.float64)

    def learn(self, x, actions, rewards):
        self.engine.learn(x.tolist(), list(map(int, actions)), list(map(float, rewards)))


class LinearPolicy:
    """Same reward-only softmax policy gradient, with a conventional linear head."""
    def __init__(self, dimensions, learning_rate=1.):
        self.weights = np.zeros((dimensions, 4), np.float32)
        self.bias = np.zeros(4, np.float32)
        self.learning_rate = learning_rate

    def predict(self, x):
        z = x @ self.weights + self.bias
        p = np.exp(z - z.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def learn(self, x, actions, rewards):
        p = self.predict(x)
        gradient = (np.eye(4, dtype=np.float32)[actions] - p) * np.asarray(rewards, np.float32)[:, None]
        self.weights += self.learning_rate * (x.T @ gradient) / len(x)
        self.bias += self.learning_rate * gradient.mean(axis=0)
