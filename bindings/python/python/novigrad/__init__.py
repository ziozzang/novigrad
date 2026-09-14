"""The Rust rate engine in-process: no HTTP server and no Python reimplementation."""
from ._native import Engine

__all__ = ['Engine']
__version__ = '0.1.1'
