import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import pyarrow as pa
import pyarrow.parquet as pq
from prepare_data import convert


class ConversionTests(unittest.TestCase):
    def fixture(self, directory, bad_index=False):
        (directory / "Completeness_783.csv").write_text(",Completed\n11,True\n22,True\n")
        pq.write_table(pa.table({
            "Presynaptic_ID": [11, 22], "Postsynaptic_ID": [22, 11],
            "Presynaptic_Index": [1 if bad_index else 0, 1], "Postsynaptic_Index": [1, 0],
            "Connectivity": [3, 2], "Excitatory": [1, -1],
            "Excitatory x Connectivity": [3, -2],
        }), directory / "Connectivity_783.parquet")

    def test_preserves_ids_counts_and_polarity(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            directory = Path(temp)
            self.fixture(directory)
            convert(directory)
            self.assertEqual((directory / "edges_783.tsv").read_text(), "11\t22\t3\t1\n22\t11\t2\t-1\n")
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["summed_synapse_counts"], 5)
            self.assertEqual(manifest["connection_rows"], 2)

    def test_rejects_mismatched_neuron_index_without_replacing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory, bad_index=True)
            target = directory / "edges_783.tsv"
            target.write_text("existing")
            with self.assertRaisesRegex(ValueError, "mapping mismatch"):
                convert(directory)
            self.assertEqual(target.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
