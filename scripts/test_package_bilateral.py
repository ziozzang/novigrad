import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from package_bilateral import MANIFEST,safe_name,sha,unpack


def fixture(path,corrupt=False):
    files={'examples/hello.py':b'print("hello")\n',
           'results/bilateral-bridge/protocol-lock.json':b'{"frozen":{}}',
           'results/bilateral-bridge/evaluation-lock.json':b'{"hashes":{}}'}
    manifest={'schema_version':1,'files':{n:{'bytes':len(b),'sha256':sha(b)} for n,b in files.items()},
              'protocol_sha256':sha(files['results/bilateral-bridge/protocol-lock.json']),
              'evaluation_sha256':sha(files['results/bilateral-bridge/evaluation-lock.json'])}
    files[MANIFEST]=json.dumps(manifest).encode()
    if corrupt:files['examples/hello.py']=b'print("wrong")\n'
    with tarfile.open(path,'w:gz') as tar:
        for name,data in files.items():
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))


class PackageTests(unittest.TestCase):
    def test_verified_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root/'x.tar.gz');result=unpack(root/'x.tar.gz',root/'new')
            self.assertEqual(result['verified_files'],3)
            self.assertEqual((root/'new/examples/hello.py').read_bytes(),b'print("hello")\n')
            with self.assertRaises(FileExistsError):unpack(root/'x.tar.gz',root/'new')
    def test_bad_hash_leaves_no_destination(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root/'x.tar.gz',True)
            with self.assertRaises(ValueError):unpack(root/'x.tar.gz',root/'new')
            self.assertFalse((root/'new').exists())
    def test_traversal_and_symlink(self):
        for value in ('../escape','/escape','a/../escape','a\\escape','a//escape','C:/escape'):
            with self.subTest(value=value),self.assertRaises(ValueError):safe_name(value)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with tarfile.open(root/'link.tar.gz','w:gz') as tar:
                info=tarfile.TarInfo('link');info.type=tarfile.SYMTYPE;info.linkname='/tmp/escape';tar.addfile(info)
            with self.assertRaises(ValueError):unpack(root/'link.tar.gz',root/'new')
            self.assertFalse((root/'new').exists())


if __name__=='__main__':unittest.main()
