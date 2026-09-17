"""Import integrity and fail-closed pairing checks; no network calls."""
import io
import unittest
import zipfile
from orn_data_pilot import read_xlsx, require_paired_rows, sha, validate_source


class ImportTests(unittest.TestCase):
    def test_changed_original_is_rejected(self):
        validate_source(b'original', 8, sha(b'original'))
        with self.assertRaises(ValueError):
            validate_source(b'changed!', 8, sha(b'original'))

    def test_pairing_does_not_follow_row_position(self):
        with self.assertRaises(ValueError):
            require_paired_rows([{'fly_id': 'Fly1', 'calcium': 3, 'spikes': 15}])
        row = dict(experiment_id='session-A', fly_id='Fly1', pair_id='trial-1',
                   calcium_source_cell='paired!B7', spike_source_cell='paired!C7')
        self.assertEqual(require_paired_rows([row]), [row])
        with self.assertRaises(ValueError):
            require_paired_rows([row, row.copy()])

    def test_xlsx_preserves_addresses_and_does_not_evaluate_formulas(self):
        data = io.BytesIO()
        ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
        with zipfile.ZipFile(data, 'w') as z:
            z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Paired" sheetId="1" r:id="r1"/></sheets></workbook>')
            z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
            z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData><row r="9"><c r="B9" t="inlineStr"><is><t>Fly7</t></is></c><c r="D9"><v>12.5</v></c><c r="E9"><f>1+2</f><v>3</v></c></row></sheetData></worksheet>')
        cells = read_xlsx(data.getvalue())['Paired']
        self.assertEqual(cells[0], {'address': 'B9', 'value': 'Fly7'})
        self.assertEqual(cells[1], {'address': 'D9', 'value': '12.5'})
        self.assertEqual(cells[2], {'address': 'E9', 'formula': True, 'value': None})


if __name__ == '__main__':
    unittest.main()
