import importlib.util
from pathlib import Path
import tempfile
import unittest

from novigrad import Engine

HAS_MLX = importlib.util.find_spec('mlx') is not None
if HAS_MLX:
    import mlx.core as mx
    import numpy as np
    from novigrad.mlx import MlxEngine
    HAS_MLX = mx.metal.is_available()


@unittest.skipUnless(HAS_MLX, 'optional mlx dependency absent')
class MetalParityTest(unittest.TestCase):
    def test_ties_signed_duplicate_edges_and_decoder(self):
        # Repeated input/plastic edges and inhibition exercise sparse-order handling.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'input.tsv').write_text('1\t10\t1\t1\n1\t10\t1\t1\n2\t10\t1\t-1\n1\t11\t1\t1\n2\t12\t1\t1\n2\t13\t1\t1\n')
            lines = [f'{hidden}\t{output}\t1\t{sign}\n' for hidden in range(10,14)
                     for output, sign in [(20,1),(21,-1),(22,1),(23,1)]]
            lines.append('10\t20\t1\t1\n')
            (root/'plastic.tsv').write_text(''.join(lines))
            cpu = Engine.from_edges(root/'input.tsv', root/'plastic.tsv', actions=2,
                                    active_fraction=.5, readout='opponent')
            cpu.set_output_actions([1,0,1,0])
            checkpoint = root/'model.safetensors'
            cpu.save(checkpoint)
            gpu = MlxEngine.load(checkpoint)
            x = [[0.,0.], [1.,1.], [1.,0.], [0.,1.], [32.,32.], [1e30,1e30]]
            expected = np.asarray(cpu.infer_batch(x))
            actual = np.asarray(gpu.infer_batch(x))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            self.assertEqual(gpu.setup['dense_matrix_bytes'], 0)
            for invalid in [[], [[-1.,0.]], [[float('nan'),0.]], [[1.]]]:
                with self.assertRaises(ValueError):
                    gpu.infer_batch(invalid)


if __name__ == '__main__':
    unittest.main()
